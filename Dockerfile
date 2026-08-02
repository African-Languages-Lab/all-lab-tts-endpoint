# Hugging Face Inference Endpoints — custom container (NOT a Space).
# Build:  docker build --platform linux/amd64 -t all-lab/all-lab-tts:ie .
# Push:   docker push <registry>/all-lab-tts:ie
# Deploy: Inference Endpoints → Custom Container → image URL, port 80
#          Set secret HF_TOKEN (read access to African-Languages-Lab/all-lab-tts-*)

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80 \
    MODEL_DIR=/repository

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" "pydantic>=2.0"

COPY alllab_tts ./alllab_tts
COPY handler.py app.py ./

# Inference Endpoints default health/probe port
EXPOSE 80

# Do not bake the 5 language packs into the image — load from Hub at runtime.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
