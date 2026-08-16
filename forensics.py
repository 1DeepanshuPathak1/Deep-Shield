import cv2
import numpy as np

FEATURE_NAMES = [
    "ai_score",
    "spectral_slope",
    "high_freq_ratio",
    "radial_peak",
    "noise_std",
    "noise_kurtosis",
    "blockiness",
    "laplacian_var",
    "edge_density",
    "saturation_mean",
    "saturation_std",
    "colour_entropy",
    "temporal_jitter",
    "temporal_flow",
]


def _grey(frame):
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def spectral_features(frame):
    grey = _grey(frame).astype(np.float32)
    grey = cv2.resize(grey, (256, 256), interpolation=cv2.INTER_AREA)
    window = np.outer(np.hanning(256), np.hanning(256))
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(grey * window)))
    spectrum = np.log1p(spectrum)

    centre = 128
    y, x = np.indices(spectrum.shape)
    radius = np.sqrt((x - centre) ** 2 + (y - centre) ** 2).astype(np.int32)
    radial = np.bincount(radius.ravel(), spectrum.ravel()) / np.maximum(
        np.bincount(radius.ravel()), 1
    )
    radial = radial[:128]

    band = radial[4:100]
    freqs = np.log(np.arange(4, 100, dtype=np.float32))
    slope = float(np.polyfit(freqs, band, 1)[0]) if band.size else 0.0

    total = float(radial[1:].sum()) or 1.0
    high_ratio = float(radial[64:].sum() / total)

    smoothed = np.convolve(radial[8:120], np.ones(5) / 5, mode="same")
    residual = radial[8:120] - smoothed
    peak = float(np.max(np.abs(residual))) if residual.size else 0.0

    return slope, high_ratio, peak


def noise_features(frame):
    grey = _grey(frame).astype(np.float32)
    denoised = cv2.medianBlur(grey.astype(np.uint8), 3).astype(np.float32)
    residual = grey - denoised
    std = float(residual.std())
    centred = residual - residual.mean()
    variance = float((centred ** 2).mean()) or 1e-6
    kurtosis = float((centred ** 4).mean() / (variance ** 2))
    return std, kurtosis


def blockiness(frame):
    grey = _grey(frame).astype(np.float32)
    h, w = grey.shape
    if h < 16 or w < 16:
        return 0.0
    vertical = np.abs(np.diff(grey, axis=1))
    horizontal = np.abs(np.diff(grey, axis=0))
    on_grid = vertical[:, 7::8].mean() + horizontal[7::8, :].mean()
    off_grid = vertical.mean() + horizontal.mean()
    return float(on_grid / (off_grid + 1e-6))


def texture_features(frame):
    grey = _grey(frame)
    lap = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    edges = cv2.Canny(grey, 80, 180)
    density = float((edges > 0).mean())
    return lap, density


def colour_features(frame):
    if frame.ndim != 3:
        return 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    hist = cv2.calcHist([hsv], [0], None, [64], [0, 180]).ravel()
    hist = hist / (hist.sum() + 1e-9)
    entropy = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
    return float(saturation.mean()), float(saturation.std()), entropy


def frame_features(frame, ai_score=0.0):
    slope, high_ratio, peak = spectral_features(frame)
    noise_std, noise_kurt = noise_features(frame)
    block = blockiness(frame)
    lap, density = texture_features(frame)
    sat_mean, sat_std, entropy = colour_features(frame)
    return {
        "ai_score": float(ai_score),
        "spectral_slope": slope,
        "high_freq_ratio": high_ratio,
        "radial_peak": peak,
        "noise_std": noise_std,
        "noise_kurtosis": noise_kurt,
        "blockiness": block,
        "laplacian_var": lap,
        "edge_density": density,
        "saturation_mean": sat_mean,
        "saturation_std": sat_std,
        "colour_entropy": entropy,
        "temporal_jitter": 0.0,
        "temporal_flow": 0.0,
    }


def temporal_features(frames):
    if len(frames) < 2:
        return 0.0, 0.0

    greys = [
        cv2.resize(_grey(f), (160, 160), interpolation=cv2.INTER_AREA) for f in frames
    ]
    diffs = []
    flows = []
    for a, b in zip(greys, greys[1:]):
        diffs.append(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()))
        flow = cv2.calcOpticalFlowFarneback(
            a, b, None, 0.5, 2, 15, 2, 5, 1.2, 0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flows.append(float(magnitude.std()))

    jitter = float(np.std(diffs)) if diffs else 0.0
    flow_var = float(np.mean(flows)) if flows else 0.0
    return jitter, flow_var


def to_vector(feature_dict):
    return np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=np.float64)
