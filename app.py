"""
FastAPI inference server for the dual-branch CNN-BiLSTM deepfake detector.
Run from project root: uvicorn backend.api:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# backend/ (contains `src/`) — must be on path so `import src.*` works
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from src.config import (  # noqa: E402
    INFERENCE_THRESHOLD,
    MAX_TIME_STEPS,
    MODEL_CHECKPOINT_PATH,
    get_inference_checkpoint_path,
)
from src.features import get_segmented_audio_features  
from src.model import dual_branch_cnn_bilstm_model  

app = FastAPI(title="Deepfake Audio Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_loaded_checkpoint: Optional[str] = None


def _features_effectively_empty(mfcc: np.ndarray, mel: np.ndarray, eps: float = 1e-6) -> bool:
    """True if decoding failed or clip is silent (matches zero fallback in features.py)."""
    return float(np.max(np.abs(mfcc))) < eps and float(np.max(np.abs(mel))) < eps


def get_model():
    global _model, _loaded_checkpoint
    ckpt = get_inference_checkpoint_path()
    if not os.path.isfile(ckpt):
        _model = None
        _loaded_checkpoint = None
        return None
    if _model is None or _loaded_checkpoint != ckpt:
        m = dual_branch_cnn_bilstm_model()
        m.load_weights(ckpt)
        _model = m
        _loaded_checkpoint = ckpt
    return _model


@app.on_event("startup")
def load_weights_on_startup():
    get_model()


@app.get("/health")
def health() -> dict[str, Any]:
    ckpt = get_inference_checkpoint_path()
    m = get_model()
    return {
        "status": "ok",
        "model_loaded": m is not None,
        "checkpoint": ckpt,
        "checkpoint_exists": os.path.isfile(ckpt),
        "default_checkpoint": MODEL_CHECKPOINT_PATH,
    }


@app.post("/predict")
def predict(
    files: List[UploadFile] = File(..., description="Up to 4 audio files (wav, mp3, ogg, flac)"),
) -> dict[str, Any]:
    if len(files) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 files allowed.")
    model = get_model()
    ckpt = get_inference_checkpoint_path()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Train first, place a .h5 file next to the project, "
                f"or set AUDIO_DEEPFAKE_MODEL_PATH. Tried: {ckpt}"
            ),
        )

    results: list[dict[str, Any]] = []
    tmp_paths: list[str] = []

    try:
        for uf in files:
            suffix = os.path.splitext(uf.filename or "audio")[1] or ".wav"
            fd, path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            tmp_paths.append(path)
            data = uf.file.read()
            if not data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Empty upload: {uf.filename or 'file'}",
                )
            with open(path, "wb") as f:
                f.write(data)

            segments = get_segmented_audio_features(path, target_length=MAX_TIME_STEPS)
            if not segments:
                raise HTTPException(
                    status_code=400,
                    detail=f"No audio segments extracted from: {uf.filename or 'file'}",
                )
            if all(_features_effectively_empty(mfcc, mel) for mfcc, mel in segments):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Could not decode usable audio (silent or unsupported format): "
                        f"{uf.filename or 'file'}. Use WAV/FLAC/OGG or MP3 with ffmpeg available."
                    ),
                )

            probs = []
            for mfcc, mel in segments:
                X_mfcc = np.expand_dims(mfcc, axis=0)
                X_mel = np.expand_dims(mel, axis=0)
                probs.append(
                    float(
                        model.predict(
                            {"input_mel": X_mel, "input_mfcc": X_mfcc},
                            verbose=0,
                        )[0][0]
                    )
                )

            prob_mean = float(np.mean(probs))
            prob_max = float(np.max(probs))
            # Use the mean score for stable decisioning, while exposing max score as context.
            prob = prob_mean
            is_deepfake = prob >= INFERENCE_THRESHOLD
            confidence = float(prob if is_deepfake else 1.0 - prob)

            results.append(
                {
                    "filename": uf.filename or "unknown",
                    "sigmoid_score": prob,
                    "segment_count": len(probs),
                    "segment_score_mean": prob_mean,
                    "segment_score_max": prob_max,
                    "is_deepfake": bool(is_deepfake),
                    "confidence": confidence,
                    "verdict": "DEEPFAKE" if is_deepfake else "AUTHENTIC",
                    "threshold_note": (
                        f">= {INFERENCE_THRESHOLD:.2f} (fake)"
                        if is_deepfake
                        else f"< {INFERENCE_THRESHOLD:.2f} (real)"
                    ),
                }
            )
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
