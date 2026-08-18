# Deep Shield

Detection of manipulated and AI-generated media, built as a major project at
Sir M. Visvesvaraya Institute of Technology (VTU).

Deep Shield does not just return a label. Every verdict is accompanied by the evidence behind
it: which checks fired, what was measured, and the rule that produced the answer.

---

## Nothing here looks for faces

The project began with a face-swap detector, and that turned out to be the wrong question.

A face-swap detector hunts for the seam where one face was composited onto another. A fully
generated scene has no seam, because nothing was composited — it is synthetic throughout. Run a
face-swap model on generated footage and it finds a perfectly coherent face and reports *Real*
with high confidence. Worse, MTCNN hallucinates faces in generated scenery: on one landscape clip
it reported 25 detections at 0.999 confidence, several of them boxes covering half the frame,
which were then scored and used to declare a face swap in a video containing no people.

Verdicts are now decided by whether footage behaves like a camera recorded it. A landscape, an
animation or an empty room is judged exactly the same way as a person talking.

| Check | Looks at | Catches |
|---|---|---|
| Motion and timing | How the picture changes between frames | Generated video, with or without people in it |
| Texture and noise | Pixel statistics of each frame | Generated stills, and generated frames in a clip |
| Voice | Acoustic properties of the audio | Cloned or synthesised speech |

Agreement between checks raises confidence. A single flag usually indicates a partial edit — real
footage given a cloned voice, for instance.

---

## How each check works

### 1. Motion and timing

Frames are sampled across the clip and compared against each other. Sixteen measurements describe
how the picture moves, none of which exist in a single frame:

| Group | Measures |
|---|---|
| Motion coherence | `warp_residual`, `warp_residual_std`, `flow_smoothness`, `flow_magnitude_var` |
| Sensor noise | `noise_level`, `noise_stability`, `noise_correlation` |
| Stability | `hf_flicker`, `edge_instability`, `sharpness_drift`, `longrange_drift` |
| Physics | `sharpness_motion_corr` |
| Global drift | `luma_drift`, `colour_drift` |
| Encoding | `residual_entropy`, `block_persistence` |

Two behave exactly as physics predicts. **`sharpness_motion_corr`**: a real shutter is open for a
fraction of a second, so fast movement smears and sharpness falls as motion rises — real footage
measures −0.34, generated footage measures 0.000, because there is no shutter to simulate.
**`noise_correlation`**: real sensor grain is regenerated every frame so consecutive frames
correlate weakly at 0.23, while generated texture is baked into the content and correlates at
0.85.

This is the strongest detector in the project, and it never touches a face.

### 2. Texture and noise

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

### Video — the primary detector

Trained on **1,756 clips**: 1,300 generated across **ten** systems (Sora, Sora 2, Veo 3,
Veo 3 fast, HunyuanVideo 1.5, LTX-2, LongCat, Wan 2.6, Kelin 2.6, FastWan 2.2) against 456 real
clips from UCF101 and DFDV.

Validation holds out an entire generator, trains on the other nine, and tests on the unseen one.

| Evaluation | Accuracy | AUC |
|---|---|---|
| Unseen generator, temporal only | **0.931** | 0.966 |
| Unseen generator, temporal + CLIP | 0.994 | 0.9996 |
| **Unseen generator AND unseen real source, temporal** | **0.860** | 0.890 |
| Unseen generator AND unseen real source, CLIP | 0.737 | 0.962 |

Quote the temporal figures. They are the more robust of the two when everything is unfamiliar —
0.860 against CLIP's 0.737 — because CLIP encodes what a scene depicts and that drifts between
domains, while the temporal measurements describe how the pixels move.

**Every clip is re-encoded to identical resolution, codec, frame rate and duration before any
measurement is taken.** Without that the classes are separable without examining content at all;
see the artefact section below.

Strongest temporal signals, after normalisation: `edge_instability` 0.735,
`noise_correlation` 0.718, `residual_entropy` 0.661, `warp_residual` 0.646,
`flow_magnitude_var` 0.633, `sharpness_motion_corr` 0.613.

### Still images

Trained on 14,594 samples pooled from six sources: ELSA_D3 (four Stable Diffusion variants),
COCO photographs, an AI-artwork corpus with its real counterpart, and frames from both halves of
the video dataset.

| Evaluation | Accuracy | AUC |
|---|---|---|
| Random 5-fold on the pooled corpus | 0.893 | 0.962 |
| **Leave-one-corpus-out (unseen generator)** | **0.556** | **0.573** |

Still images are much harder than video, because a single frame offers no motion to inspect.

**Read the second row, not the first.** The first measures how well the model recognises
material resembling what it has already seen. The second holds out an entire corpus — both its
generated and its real half — trains on the rest, and tests on the unseen one. That is what
happens when a user uploads content from a generator the model was never shown, which is the
actual use case.

Per fold:

| Held out | Accuracy | AUC |
|---|---|---|
| ELSA_D3 + COCO | 0.633 | 0.685 |
| AI artwork + real artwork | 0.477 | 0.472 |
| DFDV video | 0.559 | 0.563 |

One fold lands below chance. Cross-generator generalisation is the open research problem in this
field, and this project does not solve it.

### How the method evolved, and what each step actually proved

| Approach | In-distribution AUC | Honest AUC |
|---|---|---|
| 22 hand-crafted forensic features + gradient boosting | 0.910 | 0.806 → later 0.51 across corpora |
| CLIP ViT-B/32 embedding + linear probe, single corpus | 0.997 | 0.438 across corpora |
| CLIP + forensics, pooled across six corpora | 0.962 | **0.573** |

Two inflated results were found and discarded along the way, both by re-measuring rather than by
inspection:

**A resolution artefact.** The first image corpus mixes 32×32 thumbnails with full-size images,
and not equally per class — 73.5% of generated images were thumbnails against 70.7% of real
ones. Most of these measurements are meaningless at that size, so the model partly learned to
detect resolution, which correlated with the label. Removing thumbnails dropped the same model
from 0.910 to 0.806.

**A content artefact.** Training CLIP on ELSA_D3 against COCO reached 0.997 AUC. But ELSA_D3 is
prompt-driven artwork and COCO is everyday photography, so the probe learned *artwork versus
photograph*, not *generated versus real*. On a different corpus it scored 0.438 — below chance,
meaning it was systematically inverted.

**A codec and resolution artefact.** The first video model reported 0.9978 accuracy and a perfect
**1.0000 AUC on all ten folds**. A perfect score is a bug report, not a result. The generated
clips were 1280×720 H.264 averaging 15 seconds; the real clips were 320×240 XVID averaging 5
seconds. The classes were separable without examining a single pixel of content.

Re-encoding everything to identical 320×240 H.264 24fps 5s removed exactly the features it should
have: encoder grid persistence collapsed from 0.859 AUC to **0.506**, and noise level from 0.637
to **0.503**, both having been codec and resolution fingerprints. Edge instability held at 0.735
and noise correlation at 0.718, because those describe the content itself.

All three were caught by re-measuring on held-out data, never by reading the code. Pooling many
sources and normalising away trivial differences is what produced figures that survive contact
with unfamiliar material.

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

## Getting it running

### Quick start

Copy the whole block for your platform and paste it into a terminal. It clones the repository,
unpacks the model weights, installs the dependencies and starts the server.

Before you start you need **Python 3.11** and **ffmpeg**. If you are unsure whether you have them,
skip to [Prerequisites](#prerequisites) first — the quick start assumes both are already working.

#### Windows (PowerShell)

```powershell
git clone https://github.com/1DeepanshuPathak1/Deep-Shield.git
cd Deep-Shield

cmd /c copy /b "weights\deepshield-weights.zip.001" + "weights\deepshield-weights.zip.002" + "weights\deepshield-weights.zip.003" "weights\deepshield-weights.zip"
Expand-Archive -Path "weights\deepshield-weights.zip" -DestinationPath . -Force
Remove-Item "weights\deepshield-weights.zip"

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

flask --app app run --port 5000 --no-reload
```

If `Activate.ps1` is blocked, run this once and then repeat the line:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

#### Windows (Command Prompt)

```bat
git clone https://github.com/1DeepanshuPathak1/Deep-Shield.git
cd Deep-Shield

copy /b weights\deepshield-weights.zip.001 + weights\deepshield-weights.zip.002 + weights\deepshield-weights.zip.003 weights\deepshield-weights.zip
tar -xf weights\deepshield-weights.zip
del weights\deepshield-weights.zip

python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt

flask --app app run --port 5000 --no-reload
```

#### macOS / Linux / Git Bash

```bash
git clone https://github.com/1DeepanshuPathak1/Deep-Shield.git
cd Deep-Shield

cat weights/deepshield-weights.zip.0* > weights/deepshield-weights.zip
unzip -o weights/deepshield-weights.zip -d .
rm weights/deepshield-weights.zip

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

flask --app app run --port 5000 --no-reload
```

Then open **http://localhost:5000**.

The first request takes **30–60 seconds** while TensorFlow, MTCNN and CLIP load, and the very
first run also downloads CLIP (~600 MB) into `~/.cache/huggingface`. Later requests are fast.

---

### Prerequisites

| Requirement | Why it is needed |
|---|---|
| Python 3.11 | TensorFlow 2.18 and the pinned scikit-learn require it |
| ffmpeg and ffprobe on PATH | transcoding, audio extraction, YouTube capture |
| git | cloning |
| ~4 GB free disk | weights, dependencies and the CLIP download |

Check what you already have:

```bash
python --version     # want 3.11.x
ffmpeg -version      # want any version, just needs to run
git --version
```

**Installing ffmpeg**

```powershell
winget install Gyan.FFmpeg        # Windows
```

```bash
brew install ffmpeg               # macOS
sudo apt install ffmpeg           # Debian / Ubuntu
```

After installing on Windows, **close and reopen the terminal** so PATH refreshes, then confirm
`ffmpeg -version` runs. Without ffmpeg the site loads but video and YouTube analysis fail.

**Python 3.11 specifically.** 3.12 and 3.13 will not install TensorFlow 2.18. If `python --version`
shows something else, install 3.11 from [python.org](https://www.python.org/downloads/release/python-3119/)
and use its full path to create the virtual environment, for example:

```powershell
& "C:\Users\<you>\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
```

---

### What the weights step is doing

The four large weight files total 211 MB. GitHub refuses any single file over 100 MB, and
`model_resnet50_efficientnet_weights.h5` is 111 MB by itself, so they are compressed into one
archive and split into three parts under `weights/`. The commands above join the parts back into
one zip, extract it into the project root next to `app.py`, and delete the rebuilt zip.

After extraction the project root contains:

| File | Size | What it is |
|---|---|---|
| `model_resnet50_efficientnet_weights.h5` | 111 MB | Face-crop ensemble, from `deep_fake_audio_model.ipynb` cell 33 |
| `resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5` | 90 MB | Keras ResNet50 ImageNet weights |
| `random_forest_model.pkl` | 5.8 MB | Voice detector |
| `xgb_model.pkl` | 3.3 MB | Unused at runtime, kept for completeness |

`video_model.pkl`, `fusion_model.pkl` and `clip_probe.pkl` are committed directly and need no
unpacking.

If a part downloaded badly, compare against the published checksums:

```bash
sha256sum -c weights/SHA256SUMS.txt                                    # Linux / Git Bash
```

```powershell
Get-FileHash weights\deepshield-weights.zip.001 -Algorithm SHA256      # Windows, compare by eye
```

---

### Check it worked

With the server running, in a second terminal:

```bash
curl -o /dev/null -s -w "%{http_code}\n" http://localhost:5000/
```

`200` means the site is up. Or just open the pages:

| Page | URL | Try |
|---|---|---|
| Video | http://localhost:5000/index2 | Any clip, with or without people in it |
| Image | http://localhost:5000/image | Any still image |
| Audio | http://localhost:5000/index3 | A voice recording |
| Full Analysis | http://localhost:5000/combined | A clip with sound, or a YouTube link |
| Results | http://localhost:5000/results | Measured model performance |

---

### If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'flask'` | The virtual environment is not active. Re-run the activate line; your prompt should show `(.venv)`. |
| `Activate.ps1 cannot be loaded` | PowerShell script policy. Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| `Could not find a version that satisfies tensorflow` | Not Python 3.11. Check `python --version` and rebuild the venv with a 3.11 interpreter. |
| `ValueError: node array from the pickle has an incompatible dtype` | scikit-learn was upgraded past 1.2.2. Run `pip install -r requirements.txt` again; do not upgrade that pin. |
| `No such file or directory: 'model_resnet50_efficientnet_weights.h5'` | The weights step did not run, or extracted to the wrong place. The `.h5` files must sit beside `app.py`, not inside `weights/`. |
| `ffmpeg not found`, or video and YouTube analysis fail | ffmpeg is not on PATH. Install it, open a new terminal, confirm `ffmpeg -version` runs. |
| First analysis hangs for a minute | Expected. TensorFlow, MTCNN and CLIP load on the first request, and CLIP downloads ~600 MB on the very first run. |
| Port 5000 already in use | Use another port: `flask --app app run --port 5050 --no-reload`. |
| Edited a template but the page does not change | Flask caches templates outside debug mode. Stop the server with Ctrl+C and start it again. |

Two flags worth keeping. **`--no-reload`** matters because the reloader loads TensorFlow, MTCNN and
CLIP twice, doubling an already slow startup. **`--port`** is only needed if 5000 is taken.

## Directory structure

```
Deep-Shield/
├── app.py                    Flask routes, background job queue, verdict assembly
│
├── video_model.py            Video verdict: loads video_model.pkl
├── video_forensics.py        16 temporal measurements + frame sampling
├── clip_probe.py             CLIP ViT-B/32 embedding wrapper
├── synthetic_detector.py     Still-image generation check
├── forensics.py              22 pixel-level measurements
├── explain.py                Per-signal attribution by ablation
├── visuals.py                FFT / noise / sharpness maps, mel spectrogram
├── media_tools.py            ffmpeg + yt-dlp wrappers, YouTube URL allowlist
├── model_predict_audio.py    17 acoustic features for the voice model
├── feedback_store.py         SQLite feedback store and retraining data
│
├── video_model.pkl           Video model, trained on 10 generators   (committed)
├── fusion_model.pkl          Image generation model                  (committed)
├── clip_probe.pkl            CLIP linear probe                       (committed)
│
├── weights/                  Large weights, split for GitHub's 100 MB limit
│   ├── deepshield-weights.zip.001    88 MB
│   ├── deepshield-weights.zip.002    88 MB
│   ├── deepshield-weights.zip.003    12 MB
│   └── SHA256SUMS.txt        Checksums for parts and extracted files
├── *.h5, *.pkl               Extracted from weights/ — see step 3    (gitignored)
│
├── templates/
│   ├── base.html             Shared shell: nav, theme, scripts
│   ├── index9.html           Landing page
│   ├── index.html            Video analysis
│   ├── image.html            Image analysis
│   ├── index3.html           Audio analysis
│   ├── combined.html         Both tracks, upload or YouTube link
│   ├── index5.html           "Can You Tell?" challenge
│   ├── results.html          Challenge score
│   ├── results_graph.html    Performance charts (generated, see below)
│   └── about.html            Method and dataset documentation
│
├── static/
│   ├── css/deepshield.css    The entire stylesheet, hand-authored
│   ├── js/deepshield.js      Theme, polling, progress, evidence rendering
│   ├── mvit_crest.png        Institute crest, used as logo and favicon
│   ├── q_images/             950 face crops for the challenge page
│   ├── uploads/              Scratch space, cleared after each analysis
│   └── previews/             Transcoded playback copies, swept hourly
│
├── requirements.txt
└── README.md
```

`templates/results_graph.html` is generated rather than hand-edited. Its charts are built from
measured values by a script; editing the HTML directly will be overwritten.

### Legacy assets

`static/css/styles.css`, `static/js/{bootstrap,jquery,fancybox,parallaxie,script,wow}.js`,
`static/images/`, `static/webfonts/`, `static/project_results/` and `static/mvit_logo.png` are
left over from the original project scaffold. Nothing references them — the site uses one
stylesheet and one script — and they can be deleted.

---

## Limitations

- **The video model needs a visible face** for the face-swap check. A clip with no detectable face
  is still analysed by the generation check, but that one signal carries the verdict alone.
- **On an unseen generator the detector is close to a coin toss (0.573 AUC).** It performs well
  on material resembling its training corpora and degrades sharply outside them. Treat a verdict
  as a prompt to look closer, never as an answer.
- **Benchmark numbers here were inflated twice**, by 0.10 and 0.42 AUC respectively, and both
  were caught only by re-measuring on held-out corpora rather than by reading the code. Treat any
  single headline figure with suspicion, including 0.962.
- **The forensic measurements are shown because they are inspectable**, not because they are
  individually decisive. The strongest single feature reaches 0.694 AUC on full-resolution data.
- **The audio and face-swap models have not been stress-tested this way.** Their quoted figures
  (96.0% and 88.2%) are in-distribution, measured on held-out splits of their own datasets, and
  should be assumed to carry the same kind of optimism until tested across corpora.
- **Accuracy is dataset-bound.** Figures are measured on held-out data from these specific
  sources. Media from a different generator may score considerably lower.
- **The face-swap model false-positives on out-of-domain stills**, having been trained on video
  face crops.
- Deep Shield is a research project and a screening aid. It is not forensic proof.
