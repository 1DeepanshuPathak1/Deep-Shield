# Deep Shield

AI-based detection of manipulated audio and video, built as a major project at
Sir M. Visvesvaraya Institute of Technology (VTU).

Deep Shield does not just return a label. Every verdict is accompanied by the evidence
behind it: how many faces were inspected, how they split between authentic and manipulated,
the confidence on each one, and the exact rule that produced the final answer.

## Results

| Model | Task | Accuracy | Notes |
|---|---|---|---|
| Random Forest (100 trees, 17 features) | Audio | 96.0% | 2,637 held-out clips; Real recall 0.82, Fake recall 1.00 |
| ResNet50 + EfficientNetB0 ensemble | Video | 88.2% | 119 held-out face crops; test loss 0.394 |

The video model overfits after roughly epoch 6 — validation loss rises from that point while
training loss keeps falling. This is visible on the Results page in the app.

## Architecture

**Audio.** A clip is reduced to 17 acoustic measurements: 13 MFCCs plus spectral centroid,
chroma, zero-crossing rate and RMSE energy. A Random Forest of 100 decision trees votes on the
result, and the vote split is surfaced in the UI as a confidence signal.

**Video.** Twenty frames are sampled evenly across the clip. MTCNN locates every face, each is
resized to 100x100, and the CNN ensemble scores it. A clip is flagged once at least 5 faces are
classified as manipulated.

Analysis runs as a background job. The browser polls `/api/job/<id>` and renders live progress,
so a 12-second video analysis reports each stage as it happens rather than blocking.

## Running locally

Requires Python 3.11.

```
pip install -r requirements.txt
flask --app app run --port 5000
```

Note that `random_forest_model.pkl` is pickled with scikit-learn 1.2.2. Newer versions raise
`ValueError: node array from the pickle has an incompatible dtype` on load, so the pinned
version in `requirements.txt` matters.

## Model weights

Weights are excluded from this repository — `model_resnet50_efficientnet_weights.h5` is 112 MB,
above GitHub's 100 MB file limit. The following files must be placed in the project root before
the app will start:

| File | Size | Source |
|---|---|---|
| `model_resnet50_efficientnet_weights.h5` | 112 MB | Trained by `deep_fake_audio_model.ipynb` (cell 33) |
| `resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5` | 91 MB | Standard Keras ResNet50 ImageNet weights |
| `random_forest_model.pkl` | 6 MB | Trained by `deep_fake_audio_model.ipynb` |

## Datasets

| Dataset | Medium | Authentic | Manipulated |
|---|---|---|---|
| [SceneFake](https://www.kaggle.com/datasets/mohammedabdeldayem/scenefake) | Audio, 16 kHz WAV | 11,407 | 47,367 |
| [DFDV](https://www.kaggle.com/datasets/rajumavinmar/dfdv-video-dataset) | Video, MP4 | 49 | 50 |

SceneFake ships `train` / `dev` / `eval` splits. The model learns only from `train`; `dev` is
used during development to detect overfitting; `eval` stays sealed until the end so the reported
score reflects genuinely unseen data.

## Limitations

- The video model scores faces, not scenes. A clip with no detectable face cannot be judged.
- Reported accuracy is measured on held-out data from these specific datasets. Media from a
  different generator may score considerably lower.
- This is a research project and a screening aid, not forensic proof.
