import os
import threading

import cv2
import joblib
import numpy as np

import forensics

MODEL_NAME = "Organika/sdxl-detector"
FUSION_PATH = "fusion_model.pkl"

_lock = threading.Lock()
_processor = None
_detector = None
_torch = None
_fusion = None
_fusion_meta = {}


def _load_backbone():
    global _processor, _detector, _torch
    if _detector is not None:
        return True
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        _torch = torch
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _detector = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        _detector.eval()
        return True
    except Exception:
        _processor = None
        _detector = None
        return False


def _load_fusion():
    global _fusion, _fusion_meta
    if _fusion is not None:
        return _fusion
    if os.path.exists(FUSION_PATH):
        try:
            payload = joblib.load(FUSION_PATH)
            _fusion = payload["model"]
            _fusion_meta = {
                "accuracy": payload.get("accuracy"),
                "auc": payload.get("auc"),
                "samples": payload.get("samples"),
                "trained": payload.get("trained"),
            }
        except Exception:
            _fusion = None
    return _fusion


def reload_fusion():
    global _fusion
    with _lock:
        _fusion = None
    return _load_fusion() is not None


def fusion_info():
    _load_fusion()
    return dict(_fusion_meta)


def generation_score(frame_bgr):
    with _lock:
        if not _load_backbone():
            return None
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = _processor(images=Image.fromarray(rgb), return_tensors="pt")
        with _torch.no_grad():
            logits = _detector(**inputs).logits
        return float(_torch.softmax(logits, dim=-1)[0][0])


def analyse_frames(frames):
    if not frames:
        return None

    step = max(1, len(frames) // 8)
    sampled = frames[::step][:8]

    rows = []
    scores = []
    for frame in sampled:
        score = generation_score(frame)
        if score is not None:
            scores.append(score)
        rows.append(forensics.frame_features(frame, score if score is not None else 0.0))

    jitter, flow = forensics.temporal_features(sampled)
    for row in rows:
        row["temporal_jitter"] = jitter
        row["temporal_flow"] = flow

    matrix = np.array([forensics.to_vector(r) for r in rows], dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    averaged = matrix.mean(axis=0)

    fusion = _load_fusion()
    probability = None
    if fusion is not None:
        try:
            probability = float(fusion.predict_proba(averaged.reshape(1, -1))[0][1])
        except Exception:
            probability = None

    generation = float(np.mean(scores)) if scores else None

    if probability is not None:
        verdict = "Fake" if probability >= 0.5 else "Real"
        confidence = round(100.0 * (probability if probability >= 0.5 else 1 - probability), 1)
    elif generation is not None:
        verdict = "Fake" if generation >= 0.5 else "Real"
        confidence = round(100.0 * (generation if generation >= 0.5 else 1 - generation), 1)
    else:
        return None

    named = {name: float(v) for name, v in zip(forensics.FEATURE_NAMES, averaged)}

    drivers = []
    reference = {
        "high_freq_ratio": ("unusually strong fine detail", 1),
        "edge_density": ("very clean, low-noise edges", -1),
        "noise_std": ("sensor noise level", 1),
        "spectral_slope": ("frequency falloff", -1),
        "blockiness": ("compression grid regularity", 1),
    }
    for key, (label, direction) in reference.items():
        drivers.append(
            {
                "name": label,
                "value": round(named.get(key, 0.0), 4),
            }
        )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "probability": round(probability, 4) if probability is not None else None,
        "generationScore": round(100.0 * generation, 1) if generation is not None else None,
        "framesChecked": len(sampled),
        "features": named,
        "drivers": drivers,
        "usingFusion": fusion is not None,
        "model": fusion_info(),
    }
