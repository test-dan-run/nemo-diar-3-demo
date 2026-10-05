FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    GRADIO_ANALYTICS_ENABLED=False

# ffmpeg decodes any uploaded audio format to 16 kHz mono;
# gcc lets Triton build its CUDA launcher for torch's native GPU kernels at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Bake the weights into the image unless built with --build-arg PRELOAD_MODEL=0
ARG PRELOAD_MODEL=1
RUN if [ "$PRELOAD_MODEL" = "1" ]; then \
      python -c "from transformers import AutoModelForAudioFrameClassification, AutoProcessor; \
m='nvidia/Nemotron-3-Diarization'; AutoProcessor.from_pretrained(m); AutoModelForAudioFrameClassification.from_pretrained(m)"; \
    fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
