use super::{AudioStream, Sample, SampleFormat};
use anyhow::{Context as _, Result};
use av_codec::decoder::{Decoder, Descriptor};
use av_data::audiosample::formats;
use av_data::frame::{ArcFrame, FrameBufferConv};
use av_data::params::{AudioInfo, MediaKind};
use av_format::buffer::AccReader;
use av_format::demuxer::{Context, Event};
use av_vorbis::decoder::VORBIS_DESCR;
use libopus::decoder::OPUS_DESCR;
use matroska::demuxer::MkvDemuxer;
use std::fs::File;
use std::path::Path;

struct State {
    frame: ArcFrame,
    format: SampleFormat,
    samples: usize,
    i: usize,
}

impl State {
    fn new(frame: ArcFrame, format: SampleFormat, samples: usize) -> Self {
        Self {
            frame,
            format,
            samples,
            i: 0,
        }
    }
}

pub struct WebmContext {
    demuxer: Context<MkvDemuxer, AccReader<File>>,
    decoder: Box<dyn Decoder>,
    info: AudioInfo,
    stream_index: isize,
    state: Option<State>,
}

impl WebmContext {
    pub fn from_path(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = AccReader::with_capacity(4 * 1024, file);
        let mut demuxer = Context::new(MkvDemuxer::new(), reader);
        demuxer
            .read_headers()
            .context("Cannot parse the format headers")?;
        let (info, mut decoder, stream_index) = demuxer
            .info
            .streams
            .iter()
            .find_map(|stream| {
                let Some(MediaKind::Audio(info)) = stream.params.kind.as_ref() else {
                    log::info!("skipping non audio stream");
                    return None;
                };
                let mut decoder = match stream.params.codec_id.as_ref() {
                    Some(codec_id) if OPUS_DESCR.describe().codec == codec_id => {
                        Box::new(OPUS_DESCR.create()) as Box<dyn Decoder>
                    }
                    Some(codec_id) if VORBIS_DESCR.describe().codec == codec_id => {
                        Box::new(VORBIS_DESCR.create()) as Box<dyn Decoder>
                    }
                    Some(codec_id) => {
                        log::info!(
                            "skipping audio stream with unsupported codec id {}",
                            codec_id
                        );
                        return None;
                    }
                    None => {
                        log::info!("skipping audio stream without codec id");
                        return None;
                    }
                };
                if let Some(ref extra_data) = stream.params.extradata {
                    decoder.set_extradata(extra_data);
                }
                Some((info.clone(), decoder, stream.index as _))
            })
            .context("no supported audio stream found")?;
        decoder.configure().context("Codec configure failed")?;
        Ok(Self {
            demuxer,
            decoder,
            info,
            stream_index,
            state: None,
        })
    }

    fn next_sample(&mut self) -> Result<Option<Sample>> {
        loop {
            if let Some(state) = self.state.as_mut() {
                if state.i < state.samples {
                    let sample = match state.format {
                        SampleFormat::S16 => Sample::S16(state.frame.buf.as_slice(0)?[state.i]),
                        SampleFormat::F32 => Sample::F32(state.frame.buf.as_slice(0)?[state.i]),
                        _ => anyhow::bail!("unsupported sample format {:?}", state.format),
                    };
                    state.i += 1;
                    return Ok(Some(sample));
                } else {
                    self.state.take();
                }
            }
            match self.demuxer.read_event()? {
                Event::NewPacket(packet) => {
                    if packet.stream_index != self.stream_index {
                        continue;
                    }
                    self.decoder.send_packet(&packet)?;
                    let frame = self.decoder.receive_frame()?;
                    let info = frame.kind.get_audio_info().unwrap();
                    let format = match &*info.format {
                        &formats::S16 => SampleFormat::S16,
                        &formats::S32 => SampleFormat::S32,
                        &formats::F32 => SampleFormat::F32,
                        _ => anyhow::bail!("unsupported sample format {:?}", info.format),
                    };
                    let samples = info.samples * info.map.len();
                    self.state = Some(State::new(frame, format, samples))
                }
                Event::Eof => return Ok(None),
                _ => {}
            }
        }
    }
}

impl Iterator for WebmContext {
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_sample().transpose()
    }
}

impl AudioStream for WebmContext {
    fn sample_rate(&self) -> usize {
        self.info.rate
    }

    fn duration(&self) -> usize {
        0
    }

    fn channels(&self) -> usize {
        self.info.map.as_ref().unwrap().len()
    }
}
