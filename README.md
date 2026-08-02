---
license: apache-2.0
tags:
  - tts
  - omnivoice
  - african-languages
  - inference-endpoints
  - docker
pipeline_tag: text-to-speech
library_name: transformers
---

# ALL Lab TTS — Inference Endpoint (custom Docker)

Router package for a **Hugging Face Inference Endpoint** that serves:

- `African-Languages-Lab/all-lab-tts-hausa`
- `African-Languages-Lab/all-lab-tts-igbo`
- `African-Languages-Lab/all-lab-tts-yoruba`
- `African-Languages-Lab/all-lab-tts-twi`
- `African-Languages-Lab/all-lab-tts-ewe`

This is **not** a Space. Deploy with **Inference Endpoints → Custom Container**.

## Repo layout

| File | Role |
|------|------|
| `Dockerfile` | Custom IE container (uvicorn on port **80**) |
| `app.py` | FastAPI: `GET /health`, `POST /` |
| `handler.py` | Multi-lang OmniVoice load/generate |
| `alllab_tts/` | Inference package |
| `requirements.txt` | Pins `transformers==5.13.0` (needs `HiggsAudioV2TokenizerModel`) |

## Build & push image

On a machine with Docker (`linux/amd64`):

```bash
cd spaces/all-lab-tts-endpoint   # or a clone of this Hub repo

docker build --platform linux/amd64 -t <registry>/all-lab-tts:ie .

docker push <registry>/all-lab-tts:ie
```

`<registry>` = Docker Hub / GHCR / ECR (public, or private with registry creds in the Endpoint UI).

## Create / update Inference Endpoint

1. Open [https://endpoints.huggingface.co/all-lab/endpoints/all-lab-tts](https://endpoints.huggingface.co/all-lab/endpoints/all-lab-tts) (or **New Endpoint**).
2. **Repository:** `African-Languages-Lab/all-lab-tts-endpoint` (mounted at `/repository`).
3. **Container → Custom**:
   - Image: `<registry>/all-lab-tts:ie`
   - Port: **80**
4. Hardware: GPU (e.g. T4 / L4 / A10).
5. **Secrets:** `HF_TOKEN` = token that can read the five private `all-lab-tts-*` packs.
6. Optional: `ALL_LAB_DEFAULT_LANGUAGE=hausa`
7. Create / Update. Wait until **Running**.

## Call

```bash
export ENDPOINT_URL='https://….us-east-1.aws.endpoints.huggingface.cloud'
export HF_TOKEN=hf_...   # org token with infer access

curl -X POST "$ENDPOINT_URL" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Scale-Up-Timeout: 600" \
  -d '{"inputs":"Sannu da safe","language":"hausa"}'
```

Response includes `audio_base64` (WAV).

## Notes

- `/health` returns **503** until the handler warms; then **200**.
- Language packs are **not** in the image; they download from the Hub at runtime.
- Prefer this custom container over the default handler runtime if the stock IE image ships an old `transformers` without Higgs.
