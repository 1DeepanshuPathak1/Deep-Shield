import base64

import cv2
import numpy as np

MAP_SIDE = 320


def _encode(image):
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def _fit(frame):
    longest = max(frame.shape[:2])
    if longest > MAP_SIDE * 2:
        scale = (MAP_SIDE * 2) / float(longest)
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if longest < MAP_SIDE:
        scale = MAP_SIDE / float(longest)
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return frame


def _grey(frame):
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _stretch(plane):
    plane = plane.astype(np.float32)
    low, high = np.percentile(plane, [1, 99])
    if high - low < 1e-6:
        high = low + 1.0
    scaled = np.clip((plane - low) / (high - low), 0, 1)
    return (scaled * 255).astype(np.uint8)


def spectrum_map(frame):
    grey = cv2.resize(_grey(frame), (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    window = np.outer(np.hanning(256), np.hanning(256))
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(grey * window)))
    spectrum = np.log1p(spectrum)
    image = _stretch(spectrum)
    coloured = cv2.applyColorMap(image, cv2.COLORMAP_INFERNO)
    return _encode(cv2.resize(coloured, (MAP_SIDE, MAP_SIDE), interpolation=cv2.INTER_NEAREST))


def noise_map(frame):
    small = _fit(frame)
    grey = _grey(small).astype(np.float32)
    residual = grey - cv2.medianBlur(grey.astype(np.uint8), 3).astype(np.float32)
    amplified = np.clip(np.abs(residual) * 12.0, 0, 255).astype(np.uint8)
    coloured = cv2.applyColorMap(amplified, cv2.COLORMAP_VIRIDIS)
    return _encode(coloured)


def sharpness_map(frame):
    small = _fit(frame)
    grey = _grey(small)
    lap = np.abs(cv2.Laplacian(grey, cv2.CV_32F))
    blurred = cv2.GaussianBlur(lap, (0, 0), 6)
    coloured = cv2.applyColorMap(_stretch(blurred), cv2.COLORMAP_TURBO)
    return _encode(coloured)


def spectrogram(audio_path):
    try:
        import librosa

        y, sr = librosa.load(audio_path, sr=16000, duration=12.0)
        if y.size < 1024:
            return None
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, fmax=sr // 2)
        db = librosa.power_to_db(mel, ref=np.max)
        image = _stretch(db)
        image = np.flipud(image)
        image = cv2.resize(image, (520, 200), interpolation=cv2.INTER_LINEAR)
        coloured = cv2.applyColorMap(image, cv2.COLORMAP_MAGMA)
        return _encode(coloured)
    except Exception:
        return None


def audio_diagnostics(audio_path):
    image = spectrogram(audio_path)
    if not image:
        return []
    return [
        {
            "key": "spectrogram",
            "title": "Mel spectrogram",
            "image": image,
            "reading": (
                "Time runs left to right, pitch bottom to top, brightness is energy. Real speech "
                "shows ragged harmonic bands and noise between words. Synthesised speech often "
                "looks smoother, with unusually clean gaps and a sharp cut-off along the top edge "
                "where the vocoder stopped generating."
            ),
        }
    ]


def diagnostics(frame):
    try:
        return [
            {
                "key": "spectrum",
                "title": "Frequency spectrum",
                "image": spectrum_map(frame),
                "reading": (
                    "Energy spreading evenly to the edges, or showing a regular cross or ring, "
                    "points to generation. Photographs concentrate energy near the centre and "
                    "fade away irregularly."
                ),
            },
            {
                "key": "noise",
                "title": "Sensor noise residual",
                "image": noise_map(frame),
                "reading": (
                    "This is the grain left after removing the picture, amplified twelve times. "
                    "A real camera leaves noise spread across the whole frame. Large smooth dark "
                    "areas mean the grain is missing."
                ),
            },
            {
                "key": "sharpness",
                "title": "Sharpness distribution",
                "image": sharpness_map(frame),
                "reading": (
                    "Where the frame is sharp. A real lens focuses at one distance, so this "
                    "should vary. Uniform brightness across the whole map means everything is "
                    "equally sharp, which optics cannot do."
                ),
            },
        ]
    except Exception:
        return []
