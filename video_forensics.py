import cv2
import numpy as np

TEMPORAL_FEATURES = [
    "warp_residual",
    "warp_residual_std",
    "flow_smoothness",
    "flow_magnitude_var",
    "noise_level",
    "noise_stability",
    "noise_correlation",
    "hf_flicker",
    "edge_instability",
    "sharpness_motion_corr",
    "sharpness_drift",
    "luma_drift",
    "colour_drift",
    "longrange_drift",
    "residual_entropy",
    "block_persistence",
]

WORK_SIDE = 256


def _prepare(frames):
    prepared = []
    for frame in frames:
        if frame is None:
            continue
        h, w = frame.shape[:2]
        scale = WORK_SIDE / float(max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        prepared.append(frame)
    return prepared


def _grey(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def _noise(grey):
    denoised = cv2.medianBlur(grey.astype(np.uint8), 3).astype(np.float32)
    return grey.astype(np.float32) - denoised


def _high_frequency(grey):
    blurred = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), 2.0)
    return float(np.abs(grey.astype(np.float32) - blurred).mean())


def _warp(previous_grey, flow):
    h, w = previous_grey.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(previous_grey, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def temporal_features(frames):
    frames = _prepare(frames)
    empty = {name: 0.0 for name in TEMPORAL_FEATURES}
    if len(frames) < 3:
        return empty

    greys = [_grey(f) for f in frames]
    noises = [_noise(g) for g in greys]
    highs = [_high_frequency(g) for g in greys]
    sharps = [float(cv2.Laplacian(g, cv2.CV_32F).var()) for g in greys]
    lumas = [float(g.mean()) for g in greys]
    edges = [cv2.Canny(g, 80, 180) > 0 for g in greys]
    sats = []
    for frame in frames:
        if frame.ndim == 3:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sats.append(float(hsv[:, :, 1].mean()))
        else:
            sats.append(0.0)

    warp_errors = []
    flow_smooth = []
    flow_mags = []
    noise_corr = []
    edge_change = []
    residual_entropy = []

    for i in range(1, len(greys)):
        previous, current = greys[i - 1], greys[i]
        flow = cv2.calcOpticalFlowFarneback(
            previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        warped = _warp(previous.astype(np.float32), flow)
        residual = np.abs(warped - current.astype(np.float32))
        warp_errors.append(float(residual.mean()))

        gx = np.gradient(flow[..., 0])
        gy = np.gradient(flow[..., 1])
        smooth = float(np.mean(np.abs(gx[0]) + np.abs(gx[1]) + np.abs(gy[0]) + np.abs(gy[1])))
        flow_smooth.append(smooth)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_mags.append(float(magnitude.mean()))

        a, b = noises[i - 1].ravel(), noises[i].ravel()
        n = min(a.size, b.size)
        if n > 64 and a[:n].std() > 1e-6 and b[:n].std() > 1e-6:
            noise_corr.append(abs(float(np.corrcoef(a[:n], b[:n])[0, 1])))

        union = np.logical_or(edges[i - 1], edges[i]).sum()
        inter = np.logical_and(edges[i - 1], edges[i]).sum()
        edge_change.append(1.0 - (inter / union) if union else 0.0)

        diff = np.abs(current.astype(np.float32) - previous.astype(np.float32))
        hist, _ = np.histogram(diff, bins=32, range=(0, 255))
        p = hist / max(hist.sum(), 1)
        p = p[p > 0]
        residual_entropy.append(float(-(p * np.log2(p)).sum()))

    noise_levels = [float(n.std()) for n in noises]

    if len(flow_mags) > 2 and np.std(flow_mags) > 1e-6 and np.std(sharps[1:]) > 1e-6:
        motion_sharp = float(np.corrcoef(flow_mags, sharps[1:])[0, 1])
    else:
        motion_sharp = 0.0

    first, last = greys[0].astype(np.float32), greys[-1].astype(np.float32)
    longrange = float(np.abs(first - last).mean())

    block = []
    for i in range(1, len(greys)):
        g = greys[i].astype(np.float32)
        vertical = np.abs(np.diff(g, axis=1))
        if vertical.shape[1] >= 8:
            on = vertical[:, 7::8].mean()
            off = vertical.mean()
            block.append(float(on / (off + 1e-6)))

    return {
        "warp_residual": float(np.mean(warp_errors)) if warp_errors else 0.0,
        "warp_residual_std": float(np.std(warp_errors)) if warp_errors else 0.0,
        "flow_smoothness": float(np.mean(flow_smooth)) if flow_smooth else 0.0,
        "flow_magnitude_var": float(np.std(flow_mags)) if flow_mags else 0.0,
        "noise_level": float(np.mean(noise_levels)),
        "noise_stability": float(np.std(noise_levels)),
        "noise_correlation": float(np.mean(noise_corr)) if noise_corr else 0.0,
        "hf_flicker": float(np.std(highs)),
        "edge_instability": float(np.mean(edge_change)) if edge_change else 0.0,
        "sharpness_motion_corr": motion_sharp,
        "sharpness_drift": float(np.std(sharps) / (np.mean(sharps) + 1e-6)),
        "luma_drift": float(np.std(lumas)),
        "colour_drift": float(np.std(sats)),
        "longrange_drift": longrange,
        "residual_entropy": float(np.mean(residual_entropy)) if residual_entropy else 0.0,
        "block_persistence": float(np.mean(block)) if block else 0.0,
    }


def to_vector(features):
    return np.array([features.get(n, 0.0) for n in TEMPORAL_FEATURES], dtype=np.float64)


def sample_frames(path, count=16):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return []
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 3:
        capture.release()
        return []
    span = min(total - 1, max(count, int(total * 0.6)))
    positions = np.linspace(0, span, count, dtype=int)
    frames = []
    for pos in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()
    return frames


READABLE = {
    "warp_residual": (
        "Motion coherence",
        "Frame A is warped onto frame B along the measured motion. Real footage lines up because the world moved physically. Generated footage leaves a larger mismatch.",
    ),
    "warp_residual_std": (
        "Motion coherence stability",
        "How much that mismatch varies through the clip. Generators are inconsistent shot to shot.",
    ),
    "flow_smoothness": (
        "Motion field smoothness",
        "Real motion is spatially smooth because objects are rigid. Generated motion is patchy and discontinuous.",
    ),
    "flow_magnitude_var": (
        "Motion variability",
        "How much the amount of movement fluctuates between frames.",
    ),
    "noise_level": (
        "Sensor noise floor",
        "Every real camera leaves grain in every frame. Generated video often has almost none.",
    ),
    "noise_stability": (
        "Noise consistency",
        "Real sensor noise stays at a steady level. Generated noise appears and disappears with content.",
    ),
    "noise_correlation": (
        "Noise independence",
        "Real grain is fresh each frame, so consecutive frames correlate weakly. Baked-in synthetic texture correlates strongly.",
    ),
    "hf_flicker": (
        "Fine-detail flicker",
        "Texture that shimmers between frames. Generators redraw fine detail each frame instead of tracking it.",
    ),
    "edge_instability": (
        "Edge jitter",
        "How much detected edges fail to overlap between consecutive frames. Real edges track their objects.",
    ),
    "sharpness_motion_corr": (
        "Motion blur behaviour",
        "A real lens blurs fast movement, so sharpness drops as motion rises. Generators keep everything sharp regardless.",
    ),
    "sharpness_drift": (
        "Focus drift",
        "How much overall sharpness wanders through the clip.",
    ),
    "luma_drift": (
        "Brightness drift",
        "Global exposure wander. Generated clips drift because nothing anchors exposure.",
    ),
    "colour_drift": (
        "Colour drift",
        "Global saturation wander across the clip.",
    ),
    "longrange_drift": (
        "Scene persistence",
        "How far the last frame has departed from the first. Generators lose track of the scene over time.",
    ),
    "residual_entropy": (
        "Frame-change structure",
        "How ordered the difference between frames is. Physical motion produces structured change.",
    ),
    "block_persistence": (
        "Encoder fingerprint",
        "Consistency of the compression grid, which differs between camera pipelines and single-pass generated output.",
    ),
}
