import numpy as np

READABLE = {
    "ai_score": (
        "Diffusion-model classifier",
        "A network trained on generated images scores this frame directly.",
    ),
    "spectral_slope": (
        "Frequency falloff",
        "How quickly detail thins out from coarse to fine. Lenses fall away irregularly; generators fall away too evenly.",
    ),
    "high_freq_ratio": (
        "Fine-detail energy",
        "Share of the image held in the finest detail. Upsampled generation over-represents it.",
    ),
    "radial_peak": (
        "Spectral spikes",
        "Periodic bumps in the frequency spectrum left by fixed-size generation grids.",
    ),
    "noise_std": (
        "Sensor noise floor",
        "Every camera sensor leaves faint grain. Generated frames are often too clean.",
    ),
    "noise_kurtosis": (
        "Noise shape",
        "Real sensor noise is close to Gaussian. Synthetic residue is spikier.",
    ),
    "blockiness": (
        "Compression grid",
        "Strength of the 8x8 JPEG grid relative to ordinary detail.",
    ),
    "laplacian_var": (
        "Edge sharpness",
        "How hard edges are. Generators routinely produce edges sharper than optics can.",
    ),
    "edge_density": (
        "Edge coverage",
        "Fraction of the frame occupied by detected edges.",
    ),
    "saturation_mean": (
        "Colour intensity",
        "Average saturation. Generated imagery tends to run richer.",
    ),
    "saturation_std": (
        "Colour spread",
        "How much saturation varies across the frame.",
    ),
    "colour_entropy": (
        "Hue variety",
        "How many distinct hues are present. Generated palettes are often narrower.",
    ),
    "temporal_jitter": (
        "Frame-to-frame change",
        "How much shifts between sampled frames. Generated video drifts rather than moving coherently.",
    ),
    "temporal_flow": (
        "Motion consistency",
        "Spread of optical-flow magnitude across the frame.",
    ),
    "cfa_ratio": (
        "Sensor mosaic trace",
        "Residue of the Bayer filter a real sensor interpolates from.",
    ),
    "channel_noise_corr": (
        "Colour-channel noise link",
        "Real demosaicing correlates noise across channels.",
    ),
    "residual_energy_ratio": (
        "Channel noise balance",
        "Whether one colour channel carries far more noise than another.",
    ),
    "banding_score": (
        "Gradient banding",
        "Share of the frame that is perfectly flat, where a real gradient would dither.",
    ),
    "local_var_skew": (
        "Detail distribution",
        "Whether detail clusters in a few regions or spreads evenly.",
    ),
    "sharpness_uniformity": (
        "Depth-of-field variation",
        "Real lenses focus at one distance, so sharpness varies. Generated frames are uniformly sharp.",
    ),
    "chroma_bleed": (
        "Colour fringing",
        "Colour smearing at brightness edges, which real optics produce and generators often miss.",
    ),
    "dct_benford": (
        "Benford deviation",
        "How far DCT coefficients stray from the digit distribution photographs obey.",
    ),
}


def _percentile_of(value, quantiles):
    marks = [5, 25, 50, 75, 95]
    if value <= quantiles[0]:
        return 2.0
    if value >= quantiles[-1]:
        return 98.0
    for i in range(len(quantiles) - 1):
        low, high = quantiles[i], quantiles[i + 1]
        if low <= value <= high:
            span = high - low
            if span <= 0:
                return float(marks[i])
            frac = (value - low) / span
            return float(marks[i] + frac * (marks[i + 1] - marks[i]))
    return 50.0


def coverage(feature_names, values, stats):
    checked = 0
    outside = []
    for name in feature_names:
        info = stats.get(name) or {}
        real_q = info.get("real_q")
        ai_q = info.get("ai_q")
        if not real_q or not ai_q:
            continue
        low = min(real_q[0], ai_q[0])
        high = max(real_q[-1], ai_q[-1])
        span = (high - low) or 1.0
        value = float(values.get(name, 0.0))
        checked += 1
        if value < low or value > high:
            distance = (low - value) / span if value < low else (value - high) / span
            title = READABLE.get(name, (name.replace("_", " "), ""))[0]
            outside.append(
                {
                    "key": name,
                    "title": title,
                    "value": round(value, 4),
                    "low": round(low, 4),
                    "high": round(high, 4),
                    "distance": round(distance, 2),
                }
            )
    if not checked:
        return None
    outside.sort(key=lambda e: -e["distance"])
    share = len(outside) / checked
    return {
        "checked": checked,
        "outsideCount": len(outside),
        "share": round(100.0 * share, 1),
        "reliable": share < 0.34,
        "worst": outside[:5],
    }


def attribute(model, feature_names, values, baseline, stats, top=6):
    vector = np.array([[float(values.get(n, 0.0)) for n in feature_names]])
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        full = float(model.predict_proba(vector)[0][1])
    except Exception:
        return None, []

    base = np.array(baseline, dtype=float).reshape(1, -1)
    rows = np.repeat(vector, len(feature_names), axis=0)
    for i in range(len(feature_names)):
        rows[i, i] = base[0, i]

    try:
        ablated = model.predict_proba(rows)[:, 1]
    except Exception:
        return full, []

    findings = []
    for i, name in enumerate(feature_names):
        delta = full - float(ablated[i])
        if abs(delta) < 1e-4:
            continue
        info = stats.get(name) or {}
        value = float(vector[0, i])
        real_q = info.get("real_q")
        ai_q = info.get("ai_q")
        title, blurb = READABLE.get(name, (name.replace("_", " "), ""))

        entry = {
            "key": name,
            "title": title,
            "explanation": blurb,
            "value": round(value, 4),
            "impact": round(delta * 100.0, 2),
            "direction": "synthetic" if delta > 0 else "authentic",
            "realMedian": round(info.get("real_median", 0.0), 4),
            "aiMedian": round(info.get("ai_median", 0.0), 4),
        }
        if real_q and ai_q:
            entry["realPercentile"] = round(_percentile_of(value, real_q), 1)
            entry["aiPercentile"] = round(_percentile_of(value, ai_q), 1)
            lo = min(real_q[0], ai_q[0])
            hi = max(real_q[-1], ai_q[-1])
            span = (hi - lo) or 1.0
            entry["scale"] = {
                "position": round(100.0 * (min(max(value, lo), hi) - lo) / span, 1),
                "real": round(100.0 * (info["real_median"] - lo) / span, 1),
                "ai": round(100.0 * (info["ai_median"] - lo) / span, 1),
            }
        findings.append(entry)

    findings.sort(key=lambda e: -abs(e["impact"]))
    return full, findings[:top]


def sentence(entry):
    closer = "closer to generated imagery" if entry["direction"] == "synthetic" else "closer to camera footage"
    if "realPercentile" in entry:
        return (
            f"{entry['title']} measured {entry['value']}, against a typical "
            f"{entry['realMedian']} for real footage and {entry['aiMedian']} for generated. "
            f"That is {closer}."
        )
    return f"{entry['title']} measured {entry['value']}, which is {closer}."
