import os
import threading

import joblib
import numpy as np

import video_forensics

MODEL_PATH = "video_model.pkl"

_lock = threading.Lock()
_model = None
_meta = {}


def _load():
    global _model, _meta
    if _model is not None:
        return _model
    if os.path.exists(MODEL_PATH):
        try:
            payload = joblib.load(MODEL_PATH)
            _model = payload["model"]
            _meta = {
                "accuracy": payload.get("accuracy"),
                "auc": payload.get("auc"),
                "samples": payload.get("samples"),
                "generators": payload.get("generators"),
                "validation": payload.get("validation"),
                "trained": payload.get("trained"),
                "clipDim": payload.get("clipDim"),
                "temporalFeatures": payload.get("temporalFeatures"),
            }
        except Exception:
            _model = None
    return _model


def available():
    return _load() is not None


def info():
    _load()
    return {k: v for k, v in _meta.items() if k != "temporalFeatures"}


def reload():
    global _model
    with _lock:
        _model = None
    return _load() is not None


def _clip_vector(frames):
    try:
        import clip_probe

        if not clip_probe._load_backbone():
            return None
        step = max(1, len(frames) // 4)
        picks = frames[::step][:4]
        vectors = clip_probe.embed_batch(picks)
        if vectors is None:
            return None
        return np.asarray(vectors).mean(axis=0)
    except Exception:
        return None


def analyse(frames):
    model = _load()
    if model is None or len(frames) < 3:
        return None

    temporal = video_forensics.temporal_features(frames)
    t_vector = video_forensics.to_vector(temporal)
    c_vector = _clip_vector(frames)
    if c_vector is None:
        c_vector = np.zeros(int(_meta.get("clipDim") or 512))

    row = np.hstack([c_vector, t_vector]).reshape(1, -1)
    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        probability = float(model.predict_proba(row)[0][1])
    except Exception:
        return None

    verdict = "Fake" if probability >= 0.5 else "Real"
    confidence = round(100.0 * (probability if probability >= 0.5 else 1 - probability), 1)

    findings = []
    for name in video_forensics.TEMPORAL_FEATURES:
        title, blurb = video_forensics.READABLE.get(name, (name.replace("_", " "), ""))
        findings.append(
            {
                "key": name,
                "title": title,
                "explanation": blurb,
                "value": round(float(temporal.get(name, 0.0)), 4),
            }
        )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "probability": round(probability, 4),
        "temporal": {k: float(v) for k, v in temporal.items()},
        "signals": findings,
        "framesUsed": len(frames),
        "model": info(),
    }
