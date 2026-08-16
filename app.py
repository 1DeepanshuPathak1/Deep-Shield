import base64
import os
import random
import shutil
import threading
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from mtcnn import MTCNN
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Concatenate, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

import media_tools
from media_tools import MediaError
from model_predict_audio import extract_features, loaded_model

SAMPLE_FRAMES = 20
FAKE_FRAME_THRESHOLD = 5
INPUT_SIZE = (100, 100)
PREVIEW_TTL_SECONDS = 3600

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["PREVIEW_FOLDER"] = "static/previews/"
app.config["ALLOWED_VIDEO"] = {"mp4", "avi", "mov", "webm", "mkv"}
app.config["ALLOWED_AUDIO"] = {"wav", "mp3", "flac", "m4a", "ogg"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PREVIEW_FOLDER"], exist_ok=True)

resnet50_model = ResNet50(
    input_shape=(*INPUT_SIZE, 3),
    include_top=False,
    weights="resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
    pooling="avg",
)
efficientnet_model = EfficientNetB0(
    input_shape=(*INPUT_SIZE, 3), include_top=False, weights="imagenet", pooling="avg"
)
resnet50_model.trainable = False
efficientnet_model.trainable = False

_inputs = Input(shape=(*INPUT_SIZE, 3))
_combined = Concatenate()([resnet50_model(_inputs), efficientnet_model(_inputs)])
_x = Dense(128, activation="relu")(_combined)
_x = Dense(128, activation="relu")(_x)
_outputs = Dense(2, activation="softmax")(_x)

face_model = Model(inputs=_inputs, outputs=_outputs)
face_model.load_weights("model_resnet50_efficientnet_weights.h5")

detector = MTCNN()
model_lock = threading.Lock()

JOBS = {}
JOBS_LOCK = threading.Lock()


def create_job():
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "stage": "Queued",
            "result": None,
            "error": None,
        }
    return job_id


def update_job(job_id, **fields):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(fields)


def read_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def allowed(filename, extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions


def sweep_previews():
    now = time.time()
    folder = app.config["PREVIEW_FOLDER"]
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        try:
            if os.path.isfile(path) and now - os.path.getmtime(path) > PREVIEW_TTL_SECONDS:
                os.remove(path)
        except OSError:
            pass


def discard(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def predict_face(face_bgr):
    resized = cv2.resize(face_bgr, INPUT_SIZE)
    array = np.expand_dims(image.img_to_array(resized), axis=0)
    array = preprocess_input(array)
    with model_lock:
        scores = face_model.predict(array, verbose=0)[0]
    index = int(np.argmax(scores))
    return ("Real" if index == 1 else "Fake"), float(scores[index]) * 100.0


def encode_jpeg(frame_bgr):
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def build_playable_preview(source):
    sweep_previews()
    info = media_tools.probe(source)
    name = f"{uuid.uuid4().hex}.mp4"
    destination = os.path.join(app.config["PREVIEW_FOLDER"], name)

    if media_tools.browser_playable(info):
        shutil.copyfile(source, destination)
    else:
        media_tools.transcode_to_h264(source, destination)

    return {
        "url": "/" + destination.replace("\\", "/"),
        "converted": not media_tools.browser_playable(info),
        "sourceVideoCodec": info["videoCodec"],
        "sourceAudioCodec": info["audioCodec"],
        "hasAudio": info["hasAudio"],
        "duration": info["duration"],
        "resolution": (
            f"{info['width']}x{info['height']}" if info["width"] else None
        ),
    }


def scan_video(video_path, job_id=None, base=5, span=70):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise MediaError("The video file could not be opened.")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames < 1:
        raise MediaError("The video contains no readable frames.")

    targets = np.linspace(0, total_frames - 1, SAMPLE_FRAMES, dtype=int)
    frames = []
    real_faces = 0
    fake_faces = 0
    frames_without_face = 0
    confidences = []

    for position, target in enumerate(targets):
        if job_id:
            update_job(
                job_id,
                stage=f"Detecting faces in frame {position + 1} of {len(targets)}",
                progress=int(base + span * position / len(targets)),
            )
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ok, frame = capture.read()
        if not ok:
            continue

        detections = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not detections:
            frames_without_face += 1
            continue

        annotated = frame.copy()
        frame_labels = []
        for detection in detections:
            x, y, w, h = detection["box"]
            x, y = max(0, x), max(0, y)
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            label, confidence = predict_face(crop)
            frame_labels.append({"label": label, "confidence": round(confidence, 1)})
            confidences.append(confidence)
            if label == "Real":
                real_faces += 1
            else:
                fake_faces += 1

            colour = (76, 175, 80) if label == "Real" else (68, 68, 231)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)
            caption = f"{label} {confidence:.0f}%"
            (text_w, text_h), _ = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            box_top = max(0, y - text_h - 10)
            cv2.rectangle(
                annotated,
                (x, box_top),
                (min(x + text_w + 12, annotated.shape[1]), y),
                colour,
                -1,
            )
            cv2.putText(
                annotated,
                caption,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if not frame_labels:
            frames_without_face += 1
            continue

        fake_in_frame = sum(1 for item in frame_labels if item["label"] == "Fake")
        frames.append(
            {
                "image": encode_jpeg(annotated),
                "timestamp": round(float(target) / fps, 2),
                "faces": len(frame_labels),
                "label": "Fake" if fake_in_frame else "Real",
                "confidence": round(max(item["confidence"] for item in frame_labels), 1),
            }
        )

    capture.release()

    analysed_faces = real_faces + fake_faces
    if analysed_faces == 0:
        raise MediaError(
            "No faces were detected in this video, so the visual track cannot be analysed."
        )

    if fake_faces >= FAKE_FRAME_THRESHOLD:
        verdict = "Fake"
        reason = (
            f"{fake_faces} of the {analysed_faces} analysed faces were classified as "
            f"manipulated, which meets the alert threshold of {FAKE_FRAME_THRESHOLD}."
        )
    elif real_faces > fake_faces:
        verdict = "Real"
        reason = (
            f"{real_faces} of the {analysed_faces} analysed faces were classified as "
            f"authentic and manipulated faces stayed below the alert threshold of "
            f"{FAKE_FRAME_THRESHOLD}."
        )
    else:
        verdict = "Fake"
        reason = (
            f"Manipulated faces ({fake_faces}) matched or outnumbered authentic faces "
            f"({real_faces}) across the analysed frames."
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "fakeFaces": fake_faces,
        "realFaces": real_faces,
        "analysedFaces": analysed_faces,
        "fakeShare": round(100.0 * fake_faces / analysed_faces, 1),
        "framesSampled": len(targets),
        "framesWithFace": len(frames),
        "framesWithoutFace": frames_without_face,
        "threshold": FAKE_FRAME_THRESHOLD,
        "averageConfidence": round(sum(confidences) / len(confidences), 1),
        "peakConfidence": round(max(confidences), 1),
        "durationSeconds": round(total_frames / fps, 1),
        "resolution": f"{width}x{height}",
        "totalFrames": total_frames,
        "frames": frames,
    }


def scan_audio(audio_path):
    features = extract_features(audio_path)
    vector = np.asarray(features, dtype=float).reshape(1, -1)
    prediction = int(loaded_model.predict(vector)[0])
    verdict = "Fake" if prediction == 1 else "Real"

    probability = None
    if hasattr(loaded_model, "predict_proba"):
        probability = round(float(max(loaded_model.predict_proba(vector)[0])) * 100.0, 1)

    names = [f"MFCC {i + 1}" for i in range(13)] + [
        "Spectral Centroid",
        "Chroma",
        "Zero-Crossing Rate",
        "RMSE",
    ]
    importances = getattr(loaded_model, "feature_importances_", None)
    contributions = []
    if importances is not None:
        for index in np.argsort(importances)[::-1][:6]:
            contributions.append(
                {
                    "name": names[index],
                    "importance": round(float(importances[index]) * 100, 2),
                    "value": round(float(features[index]), 3),
                }
            )

    votes = None
    if hasattr(loaded_model, "estimators_"):
        tree_votes = [int(t.predict(vector)[0]) for t in loaded_model.estimators_]
        fake_votes = sum(tree_votes)
        votes = {
            "fake": fake_votes,
            "real": len(tree_votes) - fake_votes,
            "total": len(tree_votes),
        }

    if votes and verdict == "Fake":
        reason = f"{votes['fake']} of {votes['total']} decision trees voted that this audio is synthetic."
    elif votes:
        reason = f"{votes['real']} of {votes['total']} decision trees voted that this audio is authentic."
    else:
        reason = "The Random Forest classified this clip from its 17 acoustic features."

    return {
        "verdict": verdict,
        "confidence": probability,
        "votes": votes,
        "contributions": contributions,
        "featureCount": len(names),
        "reason": reason,
    }


def combine_verdicts(video_result, audio_result):
    flags = []
    if video_result and video_result["verdict"] == "Fake":
        flags.append("visual")
    if audio_result and audio_result["verdict"] == "Fake":
        flags.append("audio")

    if len(flags) == 2:
        return {
            "verdict": "Fake",
            "confidenceLabel": "High",
            "headline": "Both the video and the audio show signs of manipulation.",
            "reason": "The two detectors were run independently and agreed, which is the strongest signal this system can produce.",
        }
    if len(flags) == 1:
        stream = flags[0]
        other = "audio" if stream == "visual" else "visual"
        return {
            "verdict": "Fake",
            "confidenceLabel": "Mixed",
            "headline": f"The {stream} track looks manipulated, but the {other} track does not.",
            "reason": f"Only one of the two detectors raised a flag. This pattern fits a partial edit, such as a real recording with a swapped face or a genuine video given a cloned voice.",
        }
    if not video_result and not audio_result:
        return {
            "verdict": "Unknown",
            "confidenceLabel": "None",
            "headline": "Neither track could be analysed.",
            "reason": "No usable video or audio stream was found in this file.",
        }
    return {
        "verdict": "Real",
        "confidenceLabel": "High" if video_result and audio_result else "Partial",
        "headline": "Neither track shows meaningful signs of manipulation.",
        "reason": "Both detectors were run independently and neither crossed its alert threshold.",
    }


def analyse_video(job_id, video_path):
    try:
        update_job(job_id, status="running", stage="Opening video", progress=3)
        result = scan_video(video_path, job_id, base=5, span=80)
        update_job(job_id, stage="Aggregating verdict", progress=94)
        update_job(job_id, status="done", stage="Complete", progress=100, result=result)
    except Exception as exc:
        update_job(job_id, status="error", stage="Failed", progress=100, error=str(exc))
    finally:
        discard(video_path)


def analyse_audio(job_id, audio_path):
    try:
        update_job(job_id, status="running", stage="Loading waveform", progress=15)
        update_job(job_id, stage="Extracting spectral features", progress=55)
        result = scan_audio(audio_path)
        update_job(job_id, stage="Summarising evidence", progress=92)
        update_job(job_id, status="done", stage="Complete", progress=100, result=result)
    except Exception as exc:
        update_job(job_id, status="error", stage="Failed", progress=100, error=str(exc))
    finally:
        discard(audio_path)


def analyse_both(job_id, video_path, source=None):
    audio_path = video_path + ".wav"
    try:
        update_job(job_id, status="running", stage="Inspecting streams", progress=4)
        info = media_tools.probe(video_path)

        update_job(job_id, stage="Preparing playable preview", progress=10)
        preview = build_playable_preview(video_path)

        audio_result = None
        if info["hasAudio"]:
            update_job(job_id, stage="Extracting audio track", progress=18)
            media_tools.extract_audio(video_path, audio_path)
            update_job(job_id, stage="Running audio detector", progress=26)
            audio_result = scan_audio(audio_path)

        update_job(job_id, stage="Sampling frames", progress=34)
        video_result = None
        try:
            video_result = scan_video(video_path, job_id, base=36, span=56)
        except MediaError as exc:
            if not audio_result:
                raise
            video_result = None
            update_job(job_id, stage=str(exc), progress=92)

        update_job(job_id, stage="Combining verdicts", progress=95)
        result = {
            "combined": combine_verdicts(video_result, audio_result),
            "video": video_result,
            "audio": audio_result,
            "preview": preview,
            "source": source,
            "streams": {
                "hasVideo": info["hasVideo"],
                "hasAudio": info["hasAudio"],
                "videoCodec": info["videoCodec"],
                "audioCodec": info["audioCodec"],
            },
        }
        update_job(job_id, status="done", stage="Complete", progress=100, result=result)
    except Exception as exc:
        update_job(job_id, status="error", stage="Failed", progress=100, error=str(exc))
    finally:
        discard(video_path)
        discard(audio_path)


def analyse_youtube(job_id, url):
    clip_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"yt_{uuid.uuid4().hex}.mp4"
    )
    try:
        update_job(job_id, status="running", stage="Resolving link", progress=4)
        details = media_tools.describe(url)

        label = "live stream" if details["isLive"] else "video"
        update_job(
            job_id,
            stage=f"Capturing {media_tools.CLIP_SECONDS}s from {label}",
            progress=10,
        )
        media_tools.capture_clip(
            url,
            clip_path,
            seconds=media_tools.CLIP_SECONDS,
            live=details["isLive"],
        )

        analyse_both(
            job_id,
            clip_path,
            source={
                "kind": "youtube",
                "title": details["title"],
                "channel": details["channel"],
                "isLive": details["isLive"],
                "url": url,
                "clipSeconds": media_tools.CLIP_SECONDS,
            },
        )
    except Exception as exc:
        update_job(job_id, status="error", stage="Failed", progress=100, error=str(exc))
        discard(clip_path)


def start_upload_job(worker, upload, extensions, prefix):
    if not upload or upload.filename == "":
        return None, "No file was selected."
    if not allowed(upload.filename, extensions):
        return None, f"Unsupported file type: .{upload.filename.rsplit('.', 1)[-1]}"

    safe_name = f"{prefix}_{uuid.uuid4().hex}_{os.path.basename(upload.filename)}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    upload.save(path)

    job_id = create_job()
    threading.Thread(target=worker, args=(job_id, path), daemon=True).start()
    return job_id, None


@app.route("/")
def index():
    return render_template("index9.html", active="home")


@app.route("/index2")
def video_page():
    return render_template("index.html", active="video")


@app.route("/index3")
def audio_page():
    return render_template("index3.html", active="audio")


@app.route("/combined")
def combined_page():
    return render_template("combined.html", active="combined", clip=media_tools.CLIP_SECONDS)


@app.route("/about")
def about():
    return render_template("about.html", active="about")


@app.route("/api/analyze/video", methods=["POST"])
def api_analyze_video():
    job_id, error = start_upload_job(
        analyse_video, request.files.get("file"), app.config["ALLOWED_VIDEO"], "video"
    )
    if error:
        return jsonify({"error": error}), 400
    return jsonify({"jobId": job_id})


@app.route("/api/analyze/audio", methods=["POST"])
def api_analyze_audio():
    job_id, error = start_upload_job(
        analyse_audio, request.files.get("file"), app.config["ALLOWED_AUDIO"], "audio"
    )
    if error:
        return jsonify({"error": error}), 400
    return jsonify({"jobId": job_id})


@app.route("/api/analyze/combined", methods=["POST"])
def api_analyze_combined():
    job_id, error = start_upload_job(
        analyse_both, request.files.get("file"), app.config["ALLOWED_VIDEO"], "both"
    )
    if error:
        return jsonify({"error": error}), 400
    return jsonify({"jobId": job_id})


@app.route("/api/analyze/youtube", methods=["POST"])
def api_analyze_youtube():
    payload = request.get_json(silent=True) or {}
    try:
        url = media_tools.validate_youtube_url(payload.get("url"))
    except MediaError as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = create_job()
    threading.Thread(target=analyse_youtube, args=(job_id, url), daemon=True).start()
    return jsonify({"jobId": job_id})


@app.route("/api/preview", methods=["POST"])
def api_preview():
    upload = request.files.get("file")
    if not upload or upload.filename == "":
        return jsonify({"error": "No file was provided."}), 400

    temp_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"prev_{uuid.uuid4().hex}_{os.path.basename(upload.filename)}"
    )
    upload.save(temp_path)
    try:
        return jsonify(build_playable_preview(temp_path))
    except MediaError as exc:
        return jsonify({"error": str(exc)}), 415
    finally:
        discard(temp_path)


@app.route("/api/job/<job_id>")
def api_job(job_id):
    job = read_job(job_id)
    if job is None:
        return jsonify({"error": "Unknown job."}), 404
    return jsonify(job)


REAL_IMAGES_FOLDER = "static/q_images/real"
FAKE_IMAGES_FOLDER = "static/q_images/fake"


def load_quiz_images():
    real = [
        "q_images/real/" + f
        for f in os.listdir(REAL_IMAGES_FOLDER)
        if f.lower().endswith(("png", "jpg", "jpeg"))
    ]
    fake = [
        "q_images/fake/" + f
        for f in os.listdir(FAKE_IMAGES_FOLDER)
        if f.lower().endswith(("png", "jpg", "jpeg"))
    ]
    return real, fake


@app.route("/index5", methods=["GET", "POST"])
def quiz():
    if request.method == "POST":
        images = request.form.getlist("image")
        results = []
        correct = 0
        for position, path in enumerate(images):
            answer = request.form.get(f"response_{position}")
            truth = "fake" if "/fake/" in path else "real"
            is_correct = truth == answer
            correct += is_correct
            results.append(
                {
                    "image": path,
                    "truth": truth,
                    "answer": answer,
                    "correct": is_correct,
                }
            )
        return render_template(
            "results.html",
            results=results,
            correct=correct,
            total=len(results),
            active="quiz",
        )

    real, fake = load_quiz_images()
    images = real + fake
    random.shuffle(images)
    return render_template("index5.html", images=images[:12], active="quiz")


@app.route("/results")
def results_graph():
    return render_template("results_graph.html", active="results")


if __name__ == "__main__":
    app.run(debug=False, threaded=True)
