use crate::decoder::Decoder;
use anyhow::{Context, Result};
use ndarray::Array;
use ort::{Environment, ExecutionProvider, Session, SessionBuilder};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

mod audio;
mod decoder;

const MODEL: &[u8] = include_bytes!("../models/en/en_v5.onnx");
const LABELS: &str = include_str!("../models/en/en_v1_labels.json");

pub struct Silero {
    session: Session,
    decoder: Decoder,
    batch_size: usize,
    max_sequence_length: usize,
    sample_rate: usize,
}

impl Silero {
    pub fn new(model: &[u8], labels: &str) -> Result<Self> {
        let environment = Environment::builder()
            .with_name("silero")
            .with_execution_providers([ExecutionProvider::CPU(Default::default())])
            .build()?
            .into_arc();
        let session = SessionBuilder::new(&environment)?.with_model_from_memory(model)?;
        let decoder = Decoder::from_json(labels)?;
        Ok(Self {
            session,
            decoder,
            batch_size: 10,
            sample_rate: 16000,
            max_sequence_length: 172800, //12800,
        })
    }

    pub fn from_path(model: &Path, labels: &Path) -> Result<Self> {
        let model = std::fs::read(model)?;
        let labels = std::fs::read_to_string(labels)?;
        Self::new(&model, &labels)
    }

    pub fn default() -> Result<Self> {
        Self::new(MODEL, LABELS)
    }

    pub fn read_audio(&self, path: &Path) -> Result<Vec<f32>> {
        crate::audio::read_audio(path, self.sample_rate)
    }

    pub fn transcode_audio(&self, input: &Path, output: &Path) -> Result<()> {
        crate::audio::transcode_audio(input, output, self.sample_rate)
    }

    pub fn infer(&self, batch: &[Vec<f32>]) -> Result<Vec<String>> {
        let mut input = Array::zeros((batch.len(), self.max_sequence_length)).into_dyn();
        for (i, samples) in batch.iter().enumerate() {
            for (j, sample) in samples.iter().enumerate() {
                input[[i, j]] = *sample;
            }
        }
        let input_values = &input.as_standard_layout();
        let outputs = self.session.run(ort::inputs!["input" => input_values])?;
        let tensor = outputs["output"]
            .extract_tensor::<f32>()?
            .view()
            .t()
            .into_owned();
        let num_labels = tensor.slice(ndarray::s![.., 0, 0]).len();
        let num_tokens = tensor.slice(ndarray::s![0, .., 0]).len();
        let num_batches = tensor.slice(ndarray::s![0, 0, ..]).len();
        anyhow::ensure!(num_labels == self.decoder.labels().len());
        anyhow::ensure!(num_batches == batch.len());
        let mut batch = Vec::with_capacity(num_batches);
        let mut tokens = Vec::with_capacity(num_tokens);
        for i in 0..num_batches {
            for j in 0..num_tokens {
                let probs = tensor.slice(ndarray::s![.., j, i]);
                let (token, _) = probs
                    .iter()
                    .enumerate()
                    .reduce(|(ia, a), (ib, b)| if a >= b { (ia, a) } else { (ib, b) })
                    .unwrap();
                tokens.push(token);
            }
            let decoded = self.decoder.decode(&tokens)?;
            batch.push(decoded);
            tokens.clear();
        }
        Ok(batch)
    }

    fn process_batch(&self, batch: &[Vec<f32>], outputs: &[PathBuf]) -> Result<()> {
        let results = self.infer(batch)?;
        for (result, output) in results.iter().zip(outputs) {
            let mut w = BufWriter::new(OpenOptions::new().append(true).open(output)?);
            w.write_all(result.as_bytes())?;
        }
        Ok(())
    }

    pub fn stt(&self, inputs: &[PathBuf], output: &Path) -> Result<()> {
        let mut batch = Vec::with_capacity(self.batch_size);
        let mut outputs = Vec::with_capacity(self.batch_size);
        for input in inputs {
            let samples = self.read_audio(input)?;
            let basename = input
                .file_stem()
                .context("invalid input")?
                .to_str()
                .context("invalid input")?;
            let output = output.join(format!("{basename}.txt"));
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&output)?;
            for chunk in samples.chunks(self.max_sequence_length) {
                batch.push(chunk.to_vec());
                outputs.push(output.clone());
                if batch.len() == self.batch_size {
                    self.process_batch(&batch, &outputs)?;
                    batch.clear();
                    outputs.clear();
                }
            }
        }
        if !inputs.is_empty() {
            self.process_batch(&batch, &outputs)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INPUT_WEBM: &str = "example/speech_orig_opus.webm";
    const INPUT_WEBA: &str = "example/speech_orig_vorbis.weba";
    const INPUT_WAV: &str = "example/speech_orig_pcm.wav";
    const OUTPUT_WAV: &str = "example/speech_orig_transcoded.wav";
    const INPUT_TENSOR: &str = "example/input.json";
    const TOKENS: &[usize] = &[
        0, 2, 998, 157, 0, 38, 0, 998, 135, 972, 972, 969, 969, 975, 978, 978, 44, 998, 7, 998, 2,
        998, 975, 71, 972, 1, 998, 748, 0, 616, 616, 0, 0, 998, 0, 250, 983, 983, 998, 2, 998, 998,
        125, 179, 998, 998, 12, 998, 2, 998, 998, 851, 990, 990, 998, 0, 240, 321, 0, 998, 337,
        337, 337, 178, 178, 177, 177, 0, 0, 0, 998, 998, 17, 991, 975, 998, 998, 730, 986, 986,
        998, 12, 998, 461, 998, 971, 998, 78, 987, 975, 998, 20, 998, 971, 998, 998, 277, 0, 998,
        998, 0, 515, 0, 998, 998, 998, 906, 0, 975, 975, 998, 998, 998, 20, 998, 998, 473, 473, 55,
        986, 986, 998, 301, 0, 0, 998, 848, 14, 14, 998, 998, 80, 0, 0, 0, 0,
    ];
    const TEXT: &str = "the boch canoeslid on the smooth planks blew the sheet to the dark blue background it's easy to tell a deps of a well four hours of steady work faced us";
    const TEXT2: &str = "the boch canoe slid on the smooth planks blew the sheet to the dark blue background it's easy to tell a depth of a well four hours of steady work faced us";
    const TEXT3: &str = "the boch canoeslid on the smooth planks blew the sheet to the dark blue background it's easy to tell a debts of a well four hours of steady work faced us";
    const TEXT4: &str = "the boch canoeslit on the smooth planks blew the sheet to the dark blue background it's easy to tell aaddepth a well four hours of steady work faced us";

    #[test]
    fn test_decoder() -> Result<()> {
        let decoder = Decoder::from_json(LABELS.as_ref())?;
        assert_eq!(decoder.decode(&TOKENS)?, TEXT);
        Ok(())
    }

    #[test]
    fn test_inference() -> Result<()> {
        let bytes = std::fs::read(INPUT_TENSOR)?;
        let tensor: Vec<Vec<f32>> = serde_json::from_slice(&bytes)?;
        let silero = Silero::default()?;
        let result = silero.infer(&tensor)?;
        assert_eq!(result[0], TEXT);
        Ok(())
    }

    #[test]
    fn test_pipeline_wav() -> Result<()> {
        let silero = Silero::default()?;
        let samples = silero.read_audio(INPUT_WAV.as_ref())?;
        let result = silero.infer(&[samples])?;
        assert_eq!(result[0], TEXT2);
        Ok(())
    }

    #[test]
    fn test_pipeline_webm() -> Result<()> {
        let silero = Silero::default()?;
        let samples = silero.read_audio(INPUT_WEBM.as_ref())?;
        let result = silero.infer(&[samples])?;
        assert_eq!(result[0], TEXT3);
        Ok(())
    }

    #[test]
    fn test_pipeline_weba() -> Result<()> {
        let silero = Silero::default()?;
        let mut samples = silero.read_audio(INPUT_WEBA.as_ref())?;
        samples.truncate(silero.max_sequence_length);
        let result = silero.infer(&[samples])?;
        assert_eq!(result[0], TEXT4);
        Ok(())
    }

    #[test]
    #[ignore]
    fn test_transcode() -> Result<()> {
        let silero = Silero::default()?;
        silero.transcode_audio(INPUT_WEBM.as_ref(), OUTPUT_WAV.as_ref())
    }
}
