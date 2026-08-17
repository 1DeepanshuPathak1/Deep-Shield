import os
import threading

import cv2
import joblib
import numpy as np

import explain
import forensics
import visuals

BACKBONES = [
    "Organika/sdxl-detector",
]
FUSION_PATH = "fusion_model.pkl"

_lock = threading.Lock()
_backends = []
_loaded = False
_torch = None
_fusion = None
_fusion_meta = {}


def _ai_index(id2label):
    for index, label in id2label.items():
        text = str(label).lower()
        if any(k in text for k in ("artificial", "fake", "generated")) or text == "ai":
            return int(index)
    return 0


def _load_backbones():
    global _backends, _loaded, _torch
    if _loaded:
        return bool(_backends)
    _loaded = True
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        _torch = torch
        for name in BACKBONES:
            try:
                processor = AutoImageProcessor.from_pretrained(name)
                model = AutoModelForImageClassification.from_pretrained(name)
                model.eval()
                _backends.append((name, processor, model, _ai_index(model.config.id2label)))
            except Exception:
                continue
    except Exception:
        _backends = []
    return bool(_backends)


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
                "features": payload.get("features"),
                "thresholds": payload.get("thresholds"),
                "backbones": payload.get("backbones"),
                "baseline": payload.get("baseline"),
                "stats": payload.get("stats"),
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
    return {
        k: v
        for k, v in _fusion_meta.items()
        if k not in ("features", "thresholds", "baseline", "stats")
    }


def generation_scores(frame_bgr):
    result = generation_scores_batch([frame_bgr])
    return result[0] if result else {}


def generation_scores_batch(frames_bgr):
    with _lock:
        if not _load_backbones():
            return [{} for _ in frames_bgr]
        from PIL import Image

        images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr
        ]
        out = [{} for _ in frames_bgr]
        for name, processor, model, index in _backends:
            try:
                inputs = processor(images=images, return_tensors="pt")
                with _torch.no_grad():
                    logits = model(**inputs).logits
                probs = _torch.softmax(logits, dim=-1)[:, index]
                for i in range(len(frames_bgr)):
                    out[i][name] = float(probs[i])
            except Exception:
                continue
        return out


def _expected_features():
    fusion = _load_fusion()
    names = _fusion_meta.get("features")
    return names or forensics.FEATURE_NAMES


def _probability(vector_dict):
    fusion = _load_fusion()
    if fusion is None:
        return None
    names = _expected_features()
    try:
        row = np.array([[float(vector_dict.get(n, 0.0)) for n in names]])
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        return float(fusion.predict_proba(row)[0][1])
    except Exception:
        return None


TELLS = [
    (
        "laplacian_var",
        "high",
        "Edges are sharper and cleaner than a camera lens produces",
        "Edge sharpness sits in the natural camera range",
    ),
    (
        "noise_std",
        "low",
        "The sensor noise floor is unnaturally smooth for real footage",
        "Sensor noise looks like a real camera",
    ),
    (
        "high_freq_ratio",
        "high",
        "Fine detail is over-represented, a signature of upsampled generation",
        "Fine detail falls off the way optics normally do",
    ),
    (
        "spectral_slope",
        "high",
        "The frequency spectrum falls away too evenly, as diffusion output does",
        "The frequency spectrum has natural irregularity",
    ),
    (
        "temporal_jitter",
        "high",
        "Detail changes between frames in ways real motion does not",
        "Frame-to-frame change is consistent with real motion",
    ),
]

FALLBACK_THRESHOLDS = {
    "laplacian_var": 900.0,
    "noise_std": 2.2,
    "high_freq_ratio": 0.16,
    "spectral_slope": -1.1,
    "temporal_jitter": 3.0,
}


def _thresholds():
    _load_fusion()
    learned = _fusion_meta.get("thresholds") or {}
    merged = dict(FALLBACK_THRESHOLDS)
    merged.update({k: float(v) for k, v in learned.items()})
    return merged


def _tells(features):
    limits = _thresholds()
    found = []
    for key, direction, hit_text, miss_text in TELLS:
        value = float(features.get(key, 0.0))
        limit = limits[key]
        triggered = value > limit if direction == "high" else value < limit
        found.append(
            {
                "name": key.replace("_", " "),
                "triggered": bool(triggered),
                "text": hit_text if triggered else miss_text,
                "value": round(value, 3),
            }
        )
    return found


def _drivers(features):
    fusion = _load_fusion()
    if fusion is None:
        return []
    names = _expected_features()
    baseline = _fusion_meta.get("baseline")
    stats = _fusion_meta.get("stats") or {}
    if not baseline:
        return []
    _, findings = explain.attribute(fusion, names, features, baseline, stats)
    for entry in findings:
        entry["sentence"] = explain.sentence(entry)
    return findings


def analyse_image(frame_bgr):
    scores = generation_scores(frame_bgr)
    mean_score = float(np.mean(list(scores.values()))) if scores else 0.0
    features = forensics.frame_features(frame_bgr, mean_score)
    probability = _probability(features)

    if probability is None and not scores:
        return None

    effective = probability if probability is not None else mean_score
    verdict = "Fake" if effective >= 0.5 else "Real"
    confidence = round(100.0 * (effective if effective >= 0.5 else 1 - effective), 1)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "probability": round(effective, 4),
        "modelScores": {k: round(100.0 * v, 1) for k, v in scores.items()},
        "features": {k: float(v) for k, v in features.items()},
        "tells": _tells(features),
        "drivers": _drivers(features),
        "diagnostics": visuals.diagnostics(frame_bgr),
        "usingFusion": probability is not None,
        "model": fusion_info(),
    }


def analyse_frames(frames, timestamps=None):
    if not frames:
        return None

    step = max(1, len(frames) // 10)
    picked = list(range(0, len(frames), step))[:10]
    sampled = [frames[i] for i in picked]

    greys = [
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (192, 192), interpolation=cv2.INTER_AREA)
        for f in sampled
    ]

    rows = []
    per_frame = []
    model_totals = {}

    for position, frame in enumerate(sampled):
        scores = generation_scores(frame)
        for name, value in scores.items():
            model_totals.setdefault(name, []).append(value)
        mean_score = float(np.mean(list(scores.values()))) if scores else 0.0

        features = forensics.frame_features(frame, mean_score)

        if position == 0:
            delta = 0.0
        else:
            a = greys[position - 1].astype(np.float32)
            b = greys[position].astype(np.float32)
            delta = float(np.abs(a - b).mean())

        features["temporal_jitter"] = delta
        rows.append(features)

        source_index = picked[position]
        per_frame.append(
            {
                "index": source_index,
                "timestamp": (
                    timestamps[source_index] if timestamps and source_index < len(timestamps) else None
                ),
                "sharpness": round(features["laplacian_var"], 1),
                "noise": round(features["noise_std"], 3),
                "highFreq": round(features["high_freq_ratio"], 4),
                "spectralSlope": round(features["spectral_slope"], 3),
                "temporalDelta": round(delta, 2),
                "edgeDensity": round(features["edge_density"], 4),
                "modelScore": round(100.0 * mean_score, 1) if scores else None,
            }
        )

    jitter, flow = forensics.temporal_features(sampled)
    for row in rows:
        row["temporal_flow"] = flow

    averaged = {}
    for name in forensics.FEATURE_NAMES:
        averaged[name] = float(np.mean([r[name] for r in rows]))
    averaged["temporal_jitter"] = float(np.mean([p["temporalDelta"] for p in per_frame[1:]] or [0.0]))
    averaged["temporal_flow"] = flow

    probability = _probability(averaged)
    scores_mean = {k: float(np.mean(v)) for k, v in model_totals.items()}
    fallback = float(np.mean(list(scores_mean.values()))) if scores_mean else None

    if probability is None and fallback is None:
        return None

    effective = probability if probability is not None else fallback
    verdict = "Fake" if effective >= 0.5 else "Real"
    confidence = round(100.0 * (effective if effective >= 0.5 else 1 - effective), 1)

    for entry in per_frame:
        vector = dict(averaged)
        vector["laplacian_var"] = entry["sharpness"]
        vector["noise_std"] = entry["noise"]
        vector["high_freq_ratio"] = entry["highFreq"]
        vector["spectral_slope"] = entry["spectralSlope"]
        vector["edge_density"] = entry["edgeDensity"]
        vector["temporal_jitter"] = entry["temporalDelta"]
        frame_probability = _probability(vector)
        entry["probability"] = round(frame_probability, 4) if frame_probability is not None else None
        entry["verdict"] = (
            None
            if frame_probability is None
            else ("Fake" if frame_probability >= 0.5 else "Real")
        )

    strongest = max(
        range(len(per_frame)),
        key=lambda i: per_frame[i]["probability"] or 0.0,
    ) if per_frame else 0

    return {
        "verdict": verdict,
        "confidence": confidence,
        "probability": round(effective, 4),
        "framesChecked": len(sampled),
        "features": averaged,
        "perFrame": per_frame,
        "tells": _tells(averaged),
        "drivers": _drivers(averaged),
        "diagnostics": visuals.diagnostics(sampled[strongest]) if sampled else [],
        "diagnosticFrame": per_frame[strongest]["timestamp"] if per_frame else None,
        "modelScores": {k: round(100.0 * v, 1) for k, v in scores_mean.items()},
        "usingFusion": probability is not None,
        "model": fusion_info(),
    }
