# ALL Lab TTS — Inference Endpoint (custom Docker container)

Router package for a **Hugging Face Inference Endpoint**. Serves all 38
languages in the `Individual-TTS-Best-HumanEval` collection
(`all-lab/all-lab-tts-{language}`) — see `handler.py`'s `MODELS` dict for
the exact list.

This is **not** a Space. Deploy with **Inference Endpoints → Custom Container**.

## Repo layout

| File | Role |
|------|------|
| `Dockerfile` | Custom IE container (uvicorn on port **80**) |
| `app.py` | FastAPI: `GET /health`, `GET /`, `POST /` |
| `handler.py` | Multi-language OmniVoice load/generate router (`MODELS` dict = the 38 language repos) |
| `alllab_tts/` | Vendored inference package |
| `requirements.txt` | Pins `transformers==5.13.0` (needs `HiggsAudioV2TokenizerModel`) |

## Image build (automatic)

`.github/workflows/build-and-push.yml` builds and pushes on every push to
`main`. No local Docker needed.

**Image:** `ghcr.io/johnemekaeze/all-lab-tts-endpoint:latest`
**Package visibility:** must be **public** (Settings → Danger Zone → Change
visibility) at `github.com/users/johnemekaeze/packages/container/all-lab-tts-endpoint/settings`
— otherwise the Inference Endpoint's image pull fails with a 401.

## Deploy (do this in the HF UI)

1. Go to [huggingface.co/new-inference-endpoint](https://ui.endpoints.huggingface.co) (or **New Endpoint** from your HF account).
2. **Model repository:** any accessible repo works as the placeholder — this
   isn't actually used by a custom-image deployment (e.g. `all-lab/all-lab-tts-hausa`).
3. **Container type → Custom**:
   - **Image URL:** `ghcr.io/johnemekaeze/all-lab-tts-endpoint:latest`
   - **Port:** `80`
   - **Health route:** `/health`
4. **Hardware:** GPU, e.g. AWS `nvidia-t4` / `x1`.
5. **Scaling:** `min_replica=0` for scale-to-zero billing when idle.
6. **Secrets:** `HF_TOKEN` = a token with **read access to all 38**
   `all-lab/all-lab-tts-*` private model repos.
7. Optional env var: `ALL_LAB_DEFAULT_LANGUAGE=hausa` (which language warms
   at container start — defaults to `hausa`).
8. Create. Wait until state is **Running** (`/health` returns 503 until the
   default-language model finishes loading, then 200).

## Call

```bash
export ENDPOINT_URL='https://….aws.endpoints.huggingface.cloud'
export HF_TOKEN=hf_...   # token with read access to the all-lab-tts-* repos

curl -X POST "$ENDPOINT_URL" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs":"Sannu da safe","language":"hausa"}'
```

Response: `{"language", "sampling_rate", "duration_sec", "audio_base64", "format"}`.

`GET /` (once running) returns the full list of 38 supported language keys.

## Notes

- `/health` returns **503** until the handler warms; then **200**.
- Language packs are **not** baked into the image; they download from the
  Hub at runtime on first use per language, then stay cached in VRAM until a
  different language is requested (which unloads and reloads).
- This custom-container path exists specifically because HF's *default*
  Inference Endpoint runtime crashes on this model: its generic toolkit
  unconditionally imports `sentence_transformers`, which conflicts with the
  `transformers==5.13.0` this model needs for `HiggsAudioV2TokenizerModel`.
  Confirmed via repeated testing — not fixable via `requirements.txt` alone.
