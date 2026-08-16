import json
import os
import sqlite3
import threading
import time

import numpy as np

DB_PATH = "feedback.db"
RETRAIN_EVERY = 10

_lock = threading.Lock()


def _connect():
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    return connection


def init():
    with _lock, _connect() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created REAL NOT NULL,
                kind TEXT NOT NULL,
                verdict TEXT NOT NULL,
                was_correct INTEGER NOT NULL,
                true_label INTEGER NOT NULL,
                features TEXT,
                note TEXT
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS retrains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created REAL NOT NULL,
                samples INTEGER NOT NULL,
                accuracy REAL,
                auc REAL
            )
            """
        )


def record(kind, verdict, was_correct, features=None, note=None):
    predicted = 1 if verdict == "Fake" else 0
    true_label = predicted if was_correct else 1 - predicted
    with _lock, _connect() as db:
        cursor = db.execute(
            "INSERT INTO feedback (created, kind, verdict, was_correct, true_label, features, note)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                kind,
                verdict,
                1 if was_correct else 0,
                true_label,
                json.dumps(features) if features else None,
                note,
            ),
        )
        return cursor.lastrowid


def stats():
    with _lock, _connect() as db:
        row = db.execute(
            "SELECT COUNT(*) AS total, SUM(was_correct) AS correct FROM feedback"
        ).fetchone()
        total = row["total"] or 0
        correct = row["correct"] or 0
        usable = db.execute(
            "SELECT COUNT(*) AS n FROM feedback WHERE features IS NOT NULL"
        ).fetchone()["n"]
        last = db.execute(
            "SELECT created, samples, accuracy, auc FROM retrains ORDER BY id DESC LIMIT 1"
        ).fetchone()

    return {
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "agreement": round(100.0 * correct / total, 1) if total else None,
        "usable": usable,
        "untilRetrain": max(0, RETRAIN_EVERY - (usable % RETRAIN_EVERY)) if usable else RETRAIN_EVERY,
        "lastRetrain": (
            {
                "when": last["created"],
                "samples": last["samples"],
                "accuracy": last["accuracy"],
                "auc": last["auc"],
            }
            if last
            else None
        ),
    }


def labelled_features(feature_names):
    with _lock, _connect() as db:
        rows = db.execute(
            "SELECT features, true_label FROM feedback WHERE features IS NOT NULL"
        ).fetchall()

    vectors = []
    labels = []
    for row in rows:
        try:
            payload = json.loads(row["features"])
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if any(name not in payload for name in feature_names):
            continue
        vectors.append([float(payload[name]) for name in feature_names])
        labels.append(int(row["true_label"]))

    if not vectors:
        return None, None
    return np.array(vectors, dtype=np.float64), np.array(labels, dtype=np.int64)


def log_retrain(samples, accuracy, auc):
    with _lock, _connect() as db:
        db.execute(
            "INSERT INTO retrains (created, samples, accuracy, auc) VALUES (?, ?, ?, ?)",
            (time.time(), samples, accuracy, auc),
        )


init()
