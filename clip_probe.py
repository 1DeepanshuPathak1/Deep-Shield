import os
import threading

import cv2
import joblib
import numpy as np

MODEL_NAME = "openai/clip-vit-base-patch32"
PROBE_PATH = "clip_probe.pkl"

_lock = threading.Lock()
_processor = None
_model = None
_torch = None
_loaded = False
_probe = None
_probe_meta = {}


def _load_backbone():
    global _processor, _model, _torch, _loaded
    if _loaded:
        return _model is not None
    _loaded = True
    try:
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        _torch = torch
        _processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
        _model = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME)
        _model.eval()
    except Exception:
        _model = None
    return _model is not None


def embed_batch(frames_bgr):
    with _lock:
        if not _load_backbone():
            return None
        from PIL import Image

        images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr
        ]
        inputs = _processor(images=images, return_tensors="pt")
        with _torch.no_grad():
            out = _model(**inputs).image_embeds
        vectors = out.cpu().numpy()
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def embed(frame_bgr):
    result = embed_batch([frame_bgr])
    return None if result is None else result[0]


def _load_probe():
    global _probe, _probe_meta
    if _probe is not None:
        return _probe
    if os.path.exists(PROBE_PATH):
        try:
            payload = joblib.load(PROBE_PATH)
            _probe = payload["model"]
            _probe_meta = {
                "accuracy": payload.get("accuracy"),
                "auc": payload.get("auc"),
                "samples": payload.get("samples"),
                "trained": payload.get("trained"),
                "backbone": payload.get("backbone"),
                "usesForensics": payload.get("usesForensics"),
                "features": payload.get("features"),
                "heldOutAccuracy": payload.get("heldOutAccuracy"),
                "heldOutAuc": payload.get("heldOutAuc"),
                "corpora": payload.get("corpora"),
            }
        except Exception:
            _probe = None
    return _probe


def reload_probe():
    global _probe
    with _lock:
        _probe = None
    return _load_probe() is not None


def info():
    _load_probe()
    return dict(_probe_meta)


def available():
    return _load_probe() is not None


def score(frame_bgr):
    probe = _load_probe()
    if probe is None:
        return None
    vector = embed(frame_bgr)
    if vector is None:
        return None
    try:
        return float(probe.predict_proba(vector.reshape(1, -1))[0][1])
    except Exception:
        return None


def score_batch(frames_bgr):
    probe = _load_probe()
    if probe is None:
        return None
    vectors = embed_batch(frames_bgr)
    if vectors is None:
        return None
    try:
        return [float(p) for p in probe.predict_proba(vectors)[:, 1]]
    except Exception:
        return None
