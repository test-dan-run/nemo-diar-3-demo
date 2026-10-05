# Nemotron 3 Diarization — Docker demo

A small containerised service for [NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
("who spoke when", up to 8 speakers). It has:

- a REST API (FastAPI) that takes an audio clip and returns speaker segments
- a Gradio front end at `/ui`
- interactive API docs at `/docs`

The model runs through the native 🤗 Transformers integration (`transformers==5.18.0`), not the full NeMo toolkit.
The current NeMo release (3.0.0) can't load this model yet; you'd need NeMo from git main.

## Requirements

- Docker with the NVIDIA Container Toolkit (the model supports Ampere or newer GPUs)
- The model also runs on CPU (`DEVICE=cpu`). It's slower, but fine for short clips.

## Run

```bash
docker compose up --build
```

The build downloads the model weights into the image. To skip that, build with
`--build-arg PRELOAD_MODEL=0`; the weights are then downloaded on first start.

Once the container is up, open:

- Gradio UI: http://localhost:8000/ui
- API docs: http://localhost:8000/docs

## API

### `POST /diarize`

Multipart form fields:

| field          | default | description                                                   |
| -------------- | ------- | ------------------------------------------------------------- |
| `file`         | —       | Audio clip in any format ffmpeg reads (wav, mp3, flac, m4a, ...) |
| `threshold`    | `0.5`   | Speaker activity probability threshold                         |
| `min_duration` | `0.0`   | Drop segments shorter than this many seconds                   |
| `format`       | `json`  | `json` or `rttm`                                               |

```bash
curl -s -F file=@samples/diarization_example.mp3 http://localhost:8000/diarize
```

```json
{
  "duration": 97.6,
  "num_speakers": 6,
  "speakers": ["speaker_0", "speaker_1", "speaker_2", "speaker_3", "speaker_4", "speaker_5"],
  "segments": [
    {"start": 0.35, "end": 9.31, "speaker": "speaker_0"},
    {"start": 9.15, "end": 13.81, "speaker": "speaker_1"},
    {"start": 13.55, "end": 18.32, "speaker": "speaker_2"},
    ...
  ],
  "processing_time": 0.185
}
```

Speakers are numbered in the order they first speak. When two people talk at once, their segments overlap.

To get RTTM output instead of JSON:

```bash
curl -s -F file=@meeting.wav -F format=rttm http://localhost:8000/diarize
```

### `GET /health`

Returns the model ID, the device and the dtype.

## Configuration (environment variables)

| var        | default                         |                                        |
| ---------- | ------------------------------- | -------------------------------------- |
| `MODEL_ID` | `nvidia/Nemotron-3-Diarization` |                                        |
| `DEVICE`   | `auto`                          | `auto`, `cuda` or `cpu`                |
| `DTYPE`    | `float32`                       | `bfloat16` uses less memory on GPUs that support it |

## Notes

- Audio is decoded and resampled to 16 kHz mono with ffmpeg.
- Inference runs in offline mode. The model splits long recordings into chunks internally and carries a speaker
  cache between chunks, so you can send long recordings.
- The model weights are licensed under OpenMDW 1.1.
