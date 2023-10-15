use self::wav::WavContext;
use self::webm::WebmContext;
use anyhow::{Context, Result};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::path::Path;

mod wav;
mod webm;

#[derive(Clone, Copy, Debug)]
pub enum SampleFormat {
    S16,
    S32,
    F32,
}

#[derive(Clone, Copy, Debug)]
pub enum Sample {
    S16(i16),
    S32(i32),
    F32(f32),
}

impl Sample {
    pub fn to_f32(self) -> f32 {
        match self {
            Self::S16(sample) => sample as f32 / i16::MAX as f32,
            Self::S32(sample) => sample as f32 / i32::MAX as f32,
            Self::F32(sample) => sample,
        }
    }
}

impl From<Sample> for f32 {
    fn from(sample: Sample) -> Self {
        sample.to_f32()
    }
}

pub trait AudioStream: Iterator<Item = Result<Sample>> {
    fn sample_rate(&self) -> usize;
    fn duration(&self) -> usize;
    fn channels(&self) -> usize;
}

fn average_channels(stream: impl AudioStream) -> Result<Vec<f32>> {
    let mut samples = Vec::with_capacity(stream.duration());
    let channels = stream.channels();
    let mut stream = stream.map(|s| s.map(|s| s.to_f32()));
    while let Some(res) = stream.next() {
        let mut sample = res?;
        for _ in 0..(channels - 1) {
            sample += stream.next().context("invalid number of samples")??;
        }
        sample /= channels as f32;
        samples.push(sample);
    }
    Ok(samples)
}

fn resample(sample_rate: usize, target_sample_rate: usize, samples: Vec<f32>) -> Result<Vec<f32>> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(
        target_sample_rate as f64 / sample_rate as f64,
        2.0,
        params,
        samples.len(),
        1,
    )?;
    Ok(resampler
        .process(&[samples], None)?
        .into_iter()
        .next()
        .unwrap())
}

fn read_audio_stream(stream: impl AudioStream, target_sample_rate: usize) -> Result<Vec<f32>> {
    let sample_rate = stream.sample_rate();
    let samples = average_channels(stream)?;
    if sample_rate == target_sample_rate {
        return Ok(samples);
    }
    resample(sample_rate, target_sample_rate, samples)
}

pub fn read_audio(path: &Path, target_sample_rate: usize) -> Result<Vec<f32>> {
    let ext = path
        .extension()
        .context("missing extension")?
        .to_str()
        .context("invalid extension")?;
    match ext {
        "wav" => read_audio_stream(WavContext::from_path(path)?, target_sample_rate),
        "weba" | "webm" => read_audio_stream(WebmContext::from_path(path)?, target_sample_rate),
        _ => anyhow::bail!("unsupported extension {}", ext),
    }
}

pub fn transcode_audio(input: &Path, output: &Path, target_sample_rate: usize) -> Result<()> {
    let samples = read_audio(input, target_sample_rate)?;
    self::wav::write_wav(output, &samples, target_sample_rate)
}
