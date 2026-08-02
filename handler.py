"""Custom Hugging Face Inference Endpoint handler — African Languages Lab TTS.

Multi-language OmniVoice router: one endpoint, five private weight packs
(hausa, igbo, yoruba, twi, ewe). Keeps one language loaded in VRAM at a time;
switching language unloads the current pack and loads the requested one.

Payload:
    {"inputs": "Sannu da safe", "language": "hausa",
     "num_step": 24, "guidance_scale": 2.0, "denoise": true}

Response:
    {"language": "hausa", "sampling_rate": 24000, "duration_sec": 1.23,
     "audio_base64": "...", "format": "wav"}
"""

from __future__ import annotations

import base64
import gc
import io
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("all-lab-tts-endpoint")

MODELS: Dict[str, Dict[str, str]] = {
    "hausa": {"repo": "African-Languages-Lab/all-lab-tts-hausa", "language_id": "ha", "display": "Hausa"},
    "igbo": {"repo": "African-Languages-Lab/all-lab-tts-igbo", "language_id": "ig", "display": "Igbo"},
    "yoruba": {"repo": "African-Languages-Lab/all-lab-tts-yoruba", "language_id": "yo", "display": "Yoruba"},
    "twi": {"repo": "African-Languages-Lab/all-lab-tts-twi", "language_id": "tw", "display": "Twi"},
    "ewe": {"repo": "African-Languages-Lab/all-lab-tts-ewe", "language_id": "ewe", "display": "Ewe"},
}

DEFAULT_LANGUAGE = os.environ.get("ALL_LAB_DEFAULT_LANGUAGE", "hausa").strip().lower()


def _hf_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("huggingface_token")
    )


def _register_ewe() -> None:
    try:
        from alllab_tts.utils import lang_map

        lang_map.LANG_NAME_TO_ID.setdefault("ewe", "ewe")
        if hasattr(lang_map, "LANG_IDS") and isinstance(lang_map.LANG_IDS, set):
            lang_map.LANG_IDS.add("ewe")
    except Exception as exc:
        log.warning("ewe register failed: %s", exc)


class EndpointHandler:
    """Loaded once per replica by HF's Inference Endpoint runtime."""

    def __init__(self, model_dir: str = "") -> None:
        token = _hf_token()
        if token:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)

        _register_ewe()

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        self._cache: Dict[str, Any] = {"name": None, "model": None}

        # Warm the default language at container start so the *first* real
        # request after a cold start isn't also paying the model-load cost
        # on top of the container boot cost.
        try:
            self._get_model(DEFAULT_LANGUAGE)
        except Exception as exc:
            log.warning("warmup load of %s failed (will retry on first request): %s", DEFAULT_LANGUAGE, exc)

    def _unload(self) -> None:
        if self._cache.get("model") is not None:
            del self._cache["model"]
        self._cache["model"] = None
        self._cache["name"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_model(self, language: str):
        key = (language or "").strip().lower()
        if key not in MODELS:
            raise ValueError(f"Unsupported language '{language}'. Choose: {sorted(MODELS)}")
        if self._cache["name"] == key and self._cache["model"] is not None:
            return self._cache["model"], MODELS[key]

        from alllab_tts import OmniVoice

        self._unload()
        repo = MODELS[key]["repo"]
        log.info("loading %s from %s on %s", key, repo, self._device)
        model = OmniVoice.from_pretrained(
            repo,
            device_map=self._device,
            dtype=self._dtype,
            load_asr=False,
            token=_hf_token(),
        )
        model.eval()
        self._cache["name"] = key
        self._cache["model"] = model
        return model, MODELS[key]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from alllab_tts import OmniVoiceGenerationConfig

        text = (data.get("inputs") or data.get("text") or "").strip()
        if not text:
            raise ValueError("`inputs` (text to synthesize) is required")

        language = data.get("language", DEFAULT_LANGUAGE)
        parameters = data.get("parameters") or {}
        num_step = int(data.get("num_step", parameters.get("num_step", 24)))
        guidance_scale = float(data.get("guidance_scale", parameters.get("guidance_scale", 2.0)))
        denoise = bool(data.get("denoise", parameters.get("denoise", True)))
        speed = data.get("speed", parameters.get("speed"))
        duration = data.get("duration", parameters.get("duration"))

        model, meta = self._get_model(language)

        gen_cfg = OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=guidance_scale,
            denoise=denoise,
            preprocess_prompt=True,
            postprocess_output=True,
            audio_chunk_duration=12.0,
            audio_chunk_threshold=20.0,
        )
        kw: Dict[str, Any] = dict(text=text, language=meta["language_id"], generation_config=gen_cfg)
        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        audio = model.generate(**kw)
        wav = np.asarray(audio[0], dtype=np.float32)
        sr = int(model.sampling_rate)

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        raw = buf.getvalue()

        return {
            "language": language.strip().lower(),
            "sampling_rate": sr,
            "duration_sec": float(len(wav) / sr),
            "audio_base64": base64.b64encode(raw).decode("ascii"),
            "format": "wav",
        }
