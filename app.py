import base64
import os
import random
import threading
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

from model_predict_audio import extract_features, loaded_model

SAMPLE_FRAMES = 20
FAKE_FRAME_THRESHOLD = 5
INPUT_SIZE = (100, 100)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["ALLOWED_VIDEO"] = {"mp4", "avi", "mov", "webm"}
app.config["ALLOWED_AUDIO"] = {"wav", "mp3", "flac", "m4a", "ogg"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

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


def analyse_video(job_id, video_path):
    try:
        update_job(job_id, status="running", stage="Opening video", progress=3)
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError("The video file could not be opened.")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total_frames < 1:
            raise ValueError("The video contains no readable frames.")

        targets = np.linspace(0, total_frames - 1, SAMPLE_FRAMES, dtype=int)
        frames = []
        real_faces = 0
        fake_faces = 0
        frames_without_face = 0
        confidences = []

        for position, target in enumerate(targets):
            update_job(
                job_id,
                stage=f"Detecting faces in frame {position + 1} of {len(targets)}",
                progress=int(5 + 80 * position / len(targets)),
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
                    "confidence": round(
                        max(item["confidence"] for item in frame_labels), 1
                    ),
                }
            )

        capture.release()
        update_job(job_id, stage="Aggregating verdict", progress=92)

        analysed_faces = real_faces + fake_faces
        if analysed_faces == 0:
            raise ValueError(
                "No faces were detected in this video, so it cannot be analysed."
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

        result = {
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
        update_job(job_id, status="done", stage="Complete", progress=100, result=result)
    except Exception as exc:
        update_job(job_id, status="error", stage="Failed", progress=100, error=str(exc))
    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass


def analyse_audio(job_id, audio_path):
    try:
        update_job(job_id, status="running", stage="Loading waveform", progress=15)
        features = extract_features(audio_path)

        update_job(job_id, stage="Extracting spectral features", progress=55)
        vector = np.asarray(features, dtype=float).reshape(1, -1)

        update_job(job_id, stage="Running Random Forest", progress=78)
        prediction = int(loaded_model.predict(vector)[0])
        verdict = "Fake" if prediction == 1 else "Real"

        probability = None
        if hasattr(loaded_model, "predict_proba"):
            probabilities = loaded_model.predict_proba(vector)[0]
            probability = round(float(max(probabilities)) * 100.0, 1)

        update_job(job_id, stage="Summarising evidence", progress=92)
        names = [f"MFCC {i + 1}" for i in range(13)] + [
            "Spectral Centroid",
            "Chroma",
            "Zero-Crossing Rate",
            "RMSE",
        ]
        importances = getattr(loaded_model, "feature_importances_", None)
        contributions = []
        if importances is not None:
            order = np.argsort(importances)[::-1][:6]
            contributions = [
                {
                    "name": names[i],
                    "importance": round(float(importances[i]) * 100, 2),
                    "value": round(float(features[i]), 3),
                }
                for i in order
            ]

        votes = None
        if hasattr(loaded_model, "estimators_"):
            tree_votes = [int(t.predict(vector)[0]) for t in loaded_model.estimators_]
            fake_votes = sum(tree_votes)
            votes = {
                "fake": fake_votes,
                "real": len(tree_votes) - fake_votes,
                "total": len(tree_votes),
            }

        result = {
            "verdict": verdict,
            "confidence": probability,
            "votes": votes,
            "contributions": contributions,
            "featureCount": len(names),
            "reason": (
                f"{votes['fake']} of {votes['total']} decision trees voted that this clip is "
                f"synthetic." if votes and verdict == "Fake" else
                f"{votes['real']} of {votes['total']} decision trees voted that this clip is "
                f"authentic." if votes else
                "The Random Forest classified this clip from its 17 acoustic features."
            ),
        }
        update_job(job_id, status="done", stage="Complete", progress=100, result=result)
    except Exception as exc:
        update_job(job_id, status="error", stage="Failed", progress=100, error=str(exc))
    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


def start_job(worker, upload, extensions, prefix):
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


@app.route("/about")
def about():
    return render_template("about.html", active="about")


@app.route("/api/analyze/video", methods=["POST"])
def api_analyze_video():
    job_id, error = start_job(
        analyse_video, request.files.get("file"), app.config["ALLOWED_VIDEO"], "video"
    )
    if error:
        return jsonify({"error": error}), 400
    return jsonify({"jobId": job_id})


@app.route("/api/analyze/audio", methods=["POST"])
def api_analyze_audio():
    job_id, error = start_job(
        analyse_audio, request.files.get("file"), app.config["ALLOWED_AUDIO"], "audio"
    )
    if error:
        return jsonify({"error": error}), 400
    return jsonify({"jobId": job_id})


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
        responses = request.form.getlist("response")
        images = request.form.getlist("image")
        results = []
        correct = 0
        for path, answer in zip(images, responses):
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
            "results.html", results=results, correct=correct, total=len(results), active="quiz"
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
