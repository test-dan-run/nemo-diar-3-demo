"""FastAPI service exposing Nemotron 3 Diarization, with the Gradio demo mounted at /ui."""

import os
import tempfile
from contextlib import asynccontextmanager

import gradio as gr
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse, RedirectResponse

from .diarizer import SAMPLE_RATE, get_diarizer, load_audio, to_rttm
from .ui import build_ui


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the model and run one dummy pass at startup so the first request isn't slow
    # (and runtime problems such as kernel compilation surface immediately)
    diarizer = await run_in_threadpool(get_diarizer)
    await run_in_threadpool(diarizer.diarize, np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
    yield


app = FastAPI(title="Nemotron 3 Diarization API", version="1.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui")


@app.get("/health")
def health():
    d = get_diarizer()
    return {"status": "ok", "model": d.model_id, "device": d.device, "dtype": str(d.dtype).removeprefix("torch.")}


@app.post("/diarize")
async def diarize(
    file: UploadFile = File(..., description="Audio clip (wav, mp3, flac, ogg, m4a, ... anything ffmpeg reads)"),
    threshold: float = Form(0.5, ge=0.0, le=1.0, description="Speech probability threshold"),
    min_duration: float = Form(0.0, ge=0.0, description="Drop segments shorter than this (seconds)"),
    format: str = Form("json", pattern="^(json|rttm)$", description="Response format: json or rttm"),
):
    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        try:
            audio = await run_in_threadpool(load_audio, tmp.name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    result = await run_in_threadpool(get_diarizer().diarize, audio, threshold, min_duration)
    if format == "rttm":
        file_id = os.path.splitext(os.path.basename(file.filename or "audio"))[0] or "audio"
        return PlainTextResponse(to_rttm(result, file_id))
    return result


app = gr.mount_gradio_app(app, build_ui(), path="/ui")
