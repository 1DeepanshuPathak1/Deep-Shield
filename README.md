# Deep Shield

Detection of manipulated and AI-generated media, built as a major project at
Sir M. Visvesvaraya Institute of Technology (VTU).

Deep Shield does not just return a label. Every verdict is accompanied by the evidence behind
it: which checks fired, what was measured, and the rule that produced the answer.

---

## Why there are three separate detectors

A **face swap** and an **AI-generated scene** are different problems, and a detector built for
one is blind to the other.

A face-swap detector looks for the seam where one face was composited onto another. A fully
generated scene has no seam, because nothing was composited — it is synthetic all the way
through. Run a face-swap model on generated footage and it finds a perfectly coherent face and
reports *Real* with high confidence. This is a category error, not a threshold that can be tuned.

Deep Shield therefore runs three checks that never see each other's answer:

| Check | Looks at | Catches |
|---|---|---|
| Face swapping | Face crops | A real video with someone else's face pasted in |
| AI generation | Pixel statistics of the whole frame | Footage generated outright, with or without a face |
| Voice | Acoustic properties of the audio | Cloned or synthesised speech |

Agreement between checks raises confidence. A single flag usually indicates a partial edit —
real footage given a cloned voice, or a genuine recording with a swapped face.

---

## How each check works

### 1. Face swapping

Twenty frames are sampled evenly across the clip. MTCNN locates every face, each crop is resized
to 100×100, and a **ResNet50 + EfficientNetB0** ensemble scores it. The two backbones are frozen
ImageNet feature extractors; their pooled outputs are concatenated and passed through two 128-unit
dense layers to a 2-way softmax.

A clip is flagged once at least **5** faces are classified as manipulated, since a genuine video
rarely produces that many false alarms.

### 2. AI generation

This check ignores faces entirely. It measures **22 properties of the pixels** and fuses them
with a pretrained diffusion-image classifier using a gradient-boosted decision tree.

The intuition: a camera and a generative model leave different fingerprints. Real footage carries
sensor noise, lens blur that varies across the frame, and a frequency spectrum that falls away
irregularly. Generated footage tends to be too clean and too even — sharper edges than optics
produce, an unnaturally smooth noise floor, high-frequency detail over-represented because the
image was upsampled, and sharpness suspiciously uniform across the frame rather than varying
with depth.

Features, grouped by what they capture:

| Group | Features |
|---|---|
| Frequency | `spectral_slope`, `high_freq_ratio`, `radial_peak` |
| Noise | `noise_std`, `noise_kurtosis`, `channel_noise_corr`, `residual_energy_ratio` |
| Compression | `blockiness`, `dct_benford` |
| Texture | `laplacian_var`, `edge_density`, `local_var_skew`, `sharpness_uniformity` |
| Colour | `saturation_mean`, `saturation_std`, `colour_entropy`, `chroma_bleed`, `banding_score` |
| Sensor | `cfa_ratio` |
| Temporal | `temporal_jitter`, `temporal_flow` |
| Model | `ai_score` from `Organika/sdxl-detector` |

`dct_benford` tests whether the leading digits of DCT coefficients follow Benford's law, which
photographs tend to obey and generated images often do not.

### 3. Voice

A clip is reduced to 17 acoustic measurements — 13 MFCCs plus spectral centroid, chroma,
zero-crossing rate and RMSE energy. A **Random Forest of 100 trees** votes, and the vote split is
surfaced as a confidence signal: 95-to-5 means strong agreement, 55-to-45 means the clip sits near
the boundary.

---

## Measured performance

Generation detector, 5-fold cross-validated on **6,792 samples** (AI artwork, real photographs,
and frames from both halves of the video dataset):

| Signal | Accuracy | AUC |
|---|---|---|
| Pretrained classifier alone | 0.566 | 0.598 |
| Forensic features alone (logistic) | 0.738 | 0.810 |
| Forensic features alone (boosted) | 0.803 | 0.893 |
| **Fusion (boosted)** | **0.819** | **0.910** |

The forensic features carry the result — the pretrained classifier is close to chance on this
mixed image-and-video domain, because it degrades badly on compressed video frames. Gradient
boosting substantially beats logistic regression, so the relationship is non-linear.

Strongest individual features by AUC: `high_freq_ratio` 0.690, `spectral_slope` 0.627,
`dct_benford` 0.615, `laplacian_var` 0.609, `noise_std` 0.606, `sharpness_uniformity` 0.603.
Weakest, near chance: `cfa_ratio` 0.511, `channel_noise_corr` 0.508, `residual_energy_ratio` 0.506
— re-compression destroys the sensor traces these rely on.

Other detectors, on their own held-out sets:

| Model | Task | Accuracy |
|---|---|---|
| Random Forest | Audio | 96.0% on 2,637 clips (Real recall 0.82, Fake recall 1.00) |
| ResNet50 + EfficientNetB0 | Face swap | 88.2% on 119 face crops, test loss 0.394 |

The video model overfits after roughly epoch 6 — validation loss rises from that point while
training loss keeps falling. This is visible on the Results page.

---

## Learning from corrections

Every verdict can be marked correct or incorrect. The rating is stored **together with the feature
vector that produced it**, which makes it a labelled training example rather than a counter.

Every ten rated results the fusion model retrains, weighting user-supplied samples 3× above the
base training set, and hot-reloads without a restart. Retrain history is recorded so accuracy can
be tracked over time.

---

## Pages

| Route | Purpose |
|---|---|
| `/` | Landing page |
| `/index2` | Video analysis with frame-by-frame forensic evidence |
| `/image` | Still-image analysis |
| `/index3` | Audio analysis |
| `/combined` | Both tracks at once, by upload or YouTube link |
| `/index5` | "Can You Tell?" human-vs-model challenge |
| `/results` | Measured model performance, as interactive charts |
| `/about` | Method and dataset documentation |

Analysis runs as a background job: the browser polls `/api/job/<id>` and renders live progress,
so a 12-second video analysis reports each stage as it happens rather than blocking.

The `/combined` page accepts a YouTube link and captures a 30-second clip, live streams included.
Links are checked against a host allowlist before anything is fetched, so a submitted URL cannot
be aimed at internal network addresses.

---

## Datasets

| Dataset | Medium | Authentic | Generated | Used for |
|---|---|---|---|---|
| [SceneFake](https://www.kaggle.com/datasets/mohammedabdeldayem/scenefake) | Audio, 16 kHz WAV | 11,407 | 47,367 | Voice detector |
| [DFDV](https://www.kaggle.com/datasets/rajumavinmar/dfdv-video-dataset) | Video, MP4 | 49 | 50 | Face-swap detector |
| [AI-Generated vs Real Images](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets) | Images | 71,536 | 81,174 | Generation detector |

SceneFake ships `train` / `dev` / `eval` splits. The model learns only from `train`; `dev` is used
during development to detect overfitting; `eval` stays sealed until the end so the reported score
reflects genuinely unseen data.

---

## Running locally

Requires Python 3.11 and `ffmpeg` on PATH.

```
pip install -r requirements.txt
flask --app app run --port 5000
```

`random_forest_model.pkl` is pickled with scikit-learn 1.2.2. Newer versions raise
`ValueError: node array from the pickle has an incompatible dtype` on load, so the pinned version
in `requirements.txt` matters.

### Model weights

Weights are excluded from this repository — `model_resnet50_efficientnet_weights.h5` is 112 MB,
above GitHub's 100 MB file limit. Place these in the project root before starting:

| File | Size | Source |
|---|---|---|
| `model_resnet50_efficientnet_weights.h5` | 112 MB | Trained by `deep_fake_audio_model.ipynb` (cell 33) |
| `resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5` | 91 MB | Standard Keras ResNet50 ImageNet weights |
| `random_forest_model.pkl` | 6 MB | Trained by `deep_fake_audio_model.ipynb` |

`fusion_model.pkl` (the generation detector) **is** committed, since it is small.

---

## Layout

```
app.py                  Flask routes, job queue, verdict combination
forensics.py            The 22 pixel-level measurements
synthetic_detector.py   Generation check: backbone + fusion model
media_tools.py          ffmpeg and yt-dlp wrappers, URL validation
model_predict_audio.py  Audio feature extraction
feedback_store.py       Feedback persistence and retraining data
templates/              Jinja templates, all extending base.html
static/css/             Hand-authored stylesheet, no framework
```

---

## Limitations

- **The video model needs a visible face** for the face-swap check. A clip with no detectable face
  is still analysed by the generation check, but that one signal carries the verdict alone.
- **0.910 AUC is useful, not solved.** Roughly one clip in five will be misjudged. The feedback
  loop exists to close that gap on the material you actually care about.
- **Accuracy is dataset-bound.** Figures are measured on held-out data from these specific
  sources. Media from a different generator may score considerably lower.
- **The face-swap model false-positives on out-of-domain stills**, having been trained on video
  face crops.
- Deep Shield is a research project and a screening aid. It is not forensic proof.
