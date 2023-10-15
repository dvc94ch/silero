# Silero

Specify input files in wav/weba/webm/opus/vorbis format and it will transcribe them to txt
in an optional output directory.

## Dependencies
- libonnxruntime
- libopus

## AI model
- [https://github.com/snakers4/silero-models](https://github.com/sneakers4/silero-models).

Download the AI model using `models/en/download.sh`.

## What to expect
- installing libonnxruntime is a pain, but `tract` can't load the model currently
- the matroska library is poor and will fail to parse many valid webm files
- only english is currently supported

## License
Apache-2.0 + MIT