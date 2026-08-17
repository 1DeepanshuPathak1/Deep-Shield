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
    "cfa_ratio",
    "channel_noise_corr",
    "residual_energy_ratio",
    "banding_score",
    "local_var_skew",
    "sharpness_uniformity",
    "chroma_bleed",
    "dct_benford",
]


def cfa_features(frame):
    grey = _grey(frame).astype(np.float32)
    if grey.shape[0] < 32 or grey.shape[1] < 32:
        return 0.0
    denoised = cv2.GaussianBlur(grey, (3, 3), 0)
    residual = grey - denoised
    even = residual[0::2, 0::2]
    odd = residual[1::2, 1::2]
    cross = residual[0::2, 1::2]
    size = min(even.size, odd.size, cross.size)
    if size == 0:
        return 0.0
    diag = float(np.var(even.ravel()[:size]) + np.var(odd.ravel()[:size]))
    off = float(np.var(cross.ravel()[:size])) * 2.0
    return float(diag / (off + 1e-6))


def channel_noise(frame):
    if frame.ndim != 3:
        return 0.0, 0.0
    small = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
    channels = []
    for i in range(3):
        plane = small[:, :, i].astype(np.float32)
        channels.append(plane - cv2.GaussianBlur(plane, (3, 3), 0))
    flat = [c.ravel() for c in channels]
    size = min(len(f) for f in flat)
    if size < 16:
        return 0.0, 0.0
    stacked = np.vstack([f[:size] for f in flat])
    corr = np.corrcoef(stacked)
    off_diagonal = float((abs(corr[0, 1]) + abs(corr[0, 2]) + abs(corr[1, 2])) / 3.0)
    energies = [float(np.var(c)) for c in channels]
    ratio = float(max(energies) / (min(energies) + 1e-6))
    return off_diagonal, ratio


def banding(frame):
    grey = _grey(frame).astype(np.float32)
    grad_x = np.abs(np.diff(grey, axis=1))
    flat_mask = grad_x < 1.0
    if flat_mask.size == 0:
        return 0.0
    return float(flat_mask.mean())


def local_statistics(frame):
    grey = _grey(frame).astype(np.float32)
    h, w = grey.shape
    block = max(8, min(32, min(h, w) // 6))
    if h < block * 2 or w < block * 2:
        return 0.0, 0.0
    variances = []
    sharpness = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            patch = grey[y : y + block, x : x + block]
            variances.append(float(patch.var()))
            sharpness.append(float(cv2.Laplacian(patch, cv2.CV_32F).var()))
    if not variances:
        return 0.0, 0.0
    variances = np.array(variances)
    sharpness = np.array(sharpness)
    mean = variances.mean() or 1e-6
    centred = variances - variances.mean()
    denominator = (variances.std() ** 3) or 1e-6
    skew = float((centred ** 3).mean() / denominator)
    uniformity = float(sharpness.std() / (sharpness.mean() + 1e-6))
    return skew, uniformity


def chroma_bleed(frame):
    if frame.ndim != 3:
        return 0.0
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    luma_edges = cv2.Canny(ycrcb[:, :, 0], 80, 180) > 0
    cr = cv2.Laplacian(ycrcb[:, :, 1].astype(np.float32), cv2.CV_32F)
    cb = cv2.Laplacian(ycrcb[:, :, 2].astype(np.float32), cv2.CV_32F)
    chroma_energy = np.abs(cr) + np.abs(cb)
    if luma_edges.sum() < 10 or (~luma_edges).sum() < 10:
        return 0.0
    return float(chroma_energy[luma_edges].mean() / (chroma_energy[~luma_edges].mean() + 1e-6))


def dct_benford(frame):
    grey = _grey(frame).astype(np.float32)
    h, w = grey.shape
    h -= h % 8
    w -= w % 8
    if h < 8 or w < 8:
        return 0.0
    grey = grey[:h, :w]
    blocks = grey.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    step = max(1, len(blocks) // 400)
    blocks = blocks[::step]
    digits = []
    for block in blocks:
        coeffs = cv2.dct(block)
        values = np.abs(coeffs).ravel()[1:]
        values = values[values >= 1.0]
        if values.size:
            digits.append(np.floor(values / 10 ** np.floor(np.log10(values))).astype(np.int32))
    if not digits:
        return 0.0
    allc = np.concatenate(digits)
    allc = allc[(allc >= 1) & (allc <= 9)]
    if allc.size < 50:
        return 0.0
    observed = np.bincount(allc, minlength=10)[1:10].astype(np.float64)
    observed /= observed.sum()
    expected = np.log10(1 + 1 / np.arange(1, 10))
    return float(np.abs(observed - expected).sum())


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


WORK_MAX_SIDE = 512


def _bounded(frame):
    longest = max(frame.shape[:2])
    if longest <= WORK_MAX_SIDE:
        return frame
    scale = WORK_MAX_SIDE / float(longest)
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def frame_features(frame, ai_score=0.0):
    frame = _bounded(frame)
    slope, high_ratio, peak = spectral_features(frame)
    noise_std, noise_kurt = noise_features(frame)
    block = blockiness(frame)
    lap, density = texture_features(frame)
    sat_mean, sat_std, entropy = colour_features(frame)
    cfa = cfa_features(frame)
    corr, energy_ratio = channel_noise(frame)
    band = banding(frame)
    skew, uniformity = local_statistics(frame)
    bleed = chroma_bleed(frame)
    benford = dct_benford(frame)
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
        "cfa_ratio": cfa,
        "channel_noise_corr": corr,
        "residual_energy_ratio": energy_ratio,
        "banding_score": band,
        "local_var_skew": skew,
        "sharpness_uniformity": uniformity,
        "chroma_bleed": bleed,
        "dct_benford": benford,
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
