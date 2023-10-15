use super::{AudioStream, Sample};
use anyhow::Result;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub struct WavContext(WavReader<BufReader<File>>);

impl WavContext {
    pub fn from_path(path: &Path) -> Result<Self> {
        Ok(Self(WavReader::new(BufReader::new(File::open(path)?))?))
    }
}

impl AudioStream for WavContext {
    fn sample_rate(&self) -> usize {
        self.0.spec().sample_rate as _
    }

    fn duration(&self) -> usize {
        self.0.duration() as _
    }

    fn channels(&self) -> usize {
        self.0.spec().channels as _
    }
}

impl Iterator for WavContext {
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Self::Item> {
        let spec = self.0.spec();
        Some(match (spec.sample_format, spec.bits_per_sample) {
            (SampleFormat::Int, 16) => self
                .0
                .samples()
                .next()?
                .map(Sample::S16)
                .map_err(Into::into),
            (SampleFormat::Int, 32) => self
                .0
                .samples()
                .next()?
                .map(Sample::S32)
                .map_err(Into::into),
            (SampleFormat::Float, 32) => self
                .0
                .samples()
                .next()?
                .map(Sample::F32)
                .map_err(Into::into),
            _ => Err(anyhow::anyhow!("unsupported sample format")),
        })
    }
}

pub fn write_wav(path: &Path, samples: &[f32], sample_rate: usize) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate as _,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for sample in samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    Ok(())
}
