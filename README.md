# 🎙️ stt — OpenAI-Compatible Speech-to-Text API

A self-hosted, **OpenAI-compatible audio transcription API** powered by
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend)
and **FastAPI**. Drop-in replacement for OpenAI's `/v1/audio/transcriptions`
endpoint — point any OpenAI SDK/client at this server and transcribe locally.

## ✨ Features

- 🔌 **OpenAI-compatible endpoint** — `POST /v1/audio/transcriptions` with the same
  form fields (`file`, `model`, `prompt`, `response_format`, `temperature`, `language`)
- 📄 **Multiple output formats** — `json`, `text`, `verbose_json` (segments + language +
  duration), `srt` (subtitles with `HH:MM:SS,mmm` timestamps)
- 🎧 **7 audio formats** — mp3, mp4, mpeg, mpga, m4a, wav, webm
- ⚙️ **Fully env-configurable** — model size, device (`cpu`/`cuda`), compute type
  (`int8`/`float16`…), beam size, port/host
- 🔑 **Optional API-key auth** — Bearer-token guard via `ENABLE_API_KEY` / `API_KEY`
- ❤️ **Health check** — `GET /health` reports status, loaded model, and device

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- (Optional) NVIDIA GPU + CUDA for faster inference

### Installation

```bash
git clone https://github.com/suyashkrishangarg/stt.git
cd stt
pip install -r requirements.txt
```

### Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `MODEL_SIZE` | `tiny` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`… |
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `COMPUTE_TYPE` | `int8` | e.g. `int8`, `float16`, `float32` |
| `BEAM_SIZE` | `5` | Beam-search width |
| `ENABLE_API_KEY` | `false` | Set `true` to require a Bearer token |
| `API_KEY` | _(empty)_ | Expected token when auth is enabled |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Server bind address |

### Run

```bash
uvicorn openai_stt_api:app --host 0.0.0.0 --port 8000
```

Interactive docs: **http://localhost:8000/docs**


## 📖 Usage

```bash
# basic transcription (json)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1"

# subtitles output
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "response_format=srt" \
  -F "language=en" > out.srt

# with API key enabled
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@audio.mp3" \
  -F "response_format=verbose_json"
```

With the OpenAI Python SDK, just override the base URL:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
with open("audio.mp3", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper-1", file=f).text)
```

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio (multipart `file` + optional `model`, `prompt`, `response_format`, `temperature`, `language`) |
| `GET` | `/health` | Health check → `{status, model, device}` |
| `GET` | `/` | Service info + endpoint map |
| `GET` | `/docs` | Interactive Swagger UI |

## 🛠️ Tech Stack

- **Runtime:** FastAPI, Uvicorn, Pydantic
- **ASR:** faster-whisper (CTranslate2), soundfile, NumPy

## ⚠️ Notes

- Larger models (`small` → `large-v3`) are more accurate but need more RAM/VRAM;
  start with `tiny`/`base` on CPU.
- Uploaded files are written to a temp file, transcribed, then deleted.
- ⚠️ The `__main__` block references `app:app` but the file is named
  `openai_stt_api.py` — run with `uvicorn openai_stt_api:app …` as shown above.

## 📄 License

MIT — free to use and modify.
