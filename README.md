<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/GPU-not%20required-brightgreen" alt="No GPU">
  <img src="https://img.shields.io/badge/API%20Key-not%20required-brightgreen" alt="No API Key">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
  <img src="https://img.shields.io/badge/status-internal%20testing-orange" alt="Status">
</p>

<h1 align="center">🫀 FaceVitals</h1>

<p align="center">
  <strong>Non-contact health monitoring via camera</strong><br>
  Extracts heart rate, HRV, respiratory rate, stress level and dominant facial emotion from video<br>
  using remote photoplethysmography (rPPG). No GPU required.
</p>

<p align="center">
  🇧🇷 Currently under internal testing at <a href="https://namu.com.br">Namu</a> — a Brazilian health & wellness platform.<br>
  All output (terminal, dashboard, CSV) is in <strong>Brazilian Portuguese (pt-BR)</strong> by design.
</p>

---

## 🧐 What is this?

FaceVitals extracts health data by analyzing micro-variations in skin color captured by a regular camera. With each heartbeat, blood flow subtly changes the skin's color — invisible to the naked eye, but detectable by the camera.

**No sensors. No contact. No GPU. No API Keys. No external services.**

```
Camera → Frames → Face detection → ROI 72x72
                                       ↓
                             Mean RGB per frame
                                       ↓
                             POS / CHROM (BVP signal)
                                       ↓
                   ┌───────────┬────────────┬──────────────┐
                   FFT→HR    Peaks→HRV   Envelope→Resp   FER→Emotion
                   └───────────┴────────────┴──────────────┘
                                       ↓
                             Wellbeing Score
                                       ↓
                          Dashboard PNG + CSV
```

---

## 📊 What it extracts

| Indicator | Description | Accuracy (CPU) |
|-----------|-------------|:--------------:|
| 💓 Heart Rate (HR) | Beats per minute | ±0-5 bpm |
| 📈 HRV (SDNN / RMSSD) | Heart rate variability | Moderate |
| 🌬️ Respiratory Rate | Breaths per minute | Estimate |
| 🧠 Stress Level (1-5) | Based on HRV (RMSSD, SDNN, LF/HF) | Moderate |
| 😐 Dominant Emotion | Predominant facial expression | Complementary |
| ⭐ Wellbeing Score (0-100) | Combined health indicators | Indicative |

---

## ✅ Validated results

Tested with a cellphone-recorded video (good lighting), compared against **VitalScan (Namu)** and **Samsung smartwatch**:

| Indicator | FaceVitals | VitalScan | Smartwatch |
|-----------|:----------:|:---------:|:----------:|
| HR | **81 bpm** | 81 bpm | 84 bpm |
| Stress | **2 (Low)** | 2 | — |
| HRV (SDNN) | 61 ms | 94 ms | — |

> HR matched **exactly** with VitalScan. Stress level also matched. HRV has room for improvement with GPU-based neural methods.

---

## ⚡ Quick start

### Requirements

- Python 3.11+
- Webcam or cellphone camera
- **No GPU required**
- **No API Key required**
- **No account on any service required**

### Installation

```bash
# Clone the repository
git clone https://github.com/Ninjexxx/facevitals.git
cd facevitals

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements_cpu.txt

# Install FER (facial emotion detection)
pip install fer
```

---

## 🎬 How to use

### Input

FaceVitals accepts **3 input types**:

#### 1. 📹 Cellphone-recorded video (recommended)

Best quality. Record a 30+ second selfie video with good lighting and transfer it to your computer.

```bash
python mvp_rppg_v2.py --video my_video.mp4
```

#### 2. 💻 Laptop webcam

Opens the camera, shows a live preview with face detection, and captures for 30 seconds (or custom duration).

```bash
python mvp_rppg_v2.py --duration 30
```

#### 3. 📱 Cellphone as webcam

Using apps like DroidCam or Iriun Webcam. Select the camera by index.

```bash
# Use the correct camera index
python mvp_rppg_v2.py --camera 1 --duration 30
```

> **Quality ranking:** recorded video > DroidCam/streaming > laptop webcam

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--video` | Path to video file | — |
| `--camera` | Camera index (0, 1, 2...) | 0 |
| `--duration` | Capture duration in seconds | 30 |
| `--output` | Output file name | dashboard_saude.png |

---

## 📤 Output

> **🇧🇷 All output is in Brazilian Portuguese (pt-BR)** — terminal messages, dashboard labels, CSV headers, and status descriptions are in Portuguese. This is intentional, as FaceVitals is being developed and tested at [Namu](https://namu.com.br), a Brazilian company.

### 1. Terminal — Real-time summary (pt-BR)

```
============================================================
  RESUMO DE SAUDE
============================================================
  Frequencia Cardiaca:    81 bpm
  Variabilidade (SDNN):   61 ms
  Variabilidade (RMSSD):  60 ms
  Freq. Respiratoria:     10 rpm
  Nivel de Estresse:      2 - Baixo
  Emocao Dominante:       Neutro
  Score de Bem-Estar:     80/100
============================================================
```

### 2. Visual dashboard — `dashboard_saude.png`

A comprehensive chart (in pt-BR) including:
- Summary panel with all indicators
- BVP signal (blood volume pulse) with detected heartbeats
- Heart rate frequency spectrum
- Respiratory signal
- Inter-beat intervals (IBI)
- Stress and emotion indicators
- Wellbeing score

### 3. Exported data — `dashboard_saude.csv`

CSV with measurement history (headers in pt-BR). Each run appends a new row.

| Column | Example |
|--------|---------|
| timestamp | 2026-03-27T11:16:02 |
| method | CHROM |
| hr_bpm | 81.0 |
| rr_rpm | 10.0 |
| stress | 2 - Baixo |
| sdnn_ms | 61.0 |
| rmssd_ms | 60.0 |
| emotion | Neutro |
| wellbeing_score | 80 |

---

## 🔬 How it works

### Vital signs extraction

1. **Face detection** — Haar Cascade (OpenCV) locates the face in each frame
2. **ROI** — Crops and resizes to 72x72 pixels
3. **RGB signal** — Computes the mean R, G, B values per frame
4. **rPPG algorithms** — POS and CHROM process the RGB signal to isolate the blood volume pulse (BVP)
5. **Auto-selection** — Compares SNR of both methods and uses the best one

### Derived indicators

| Indicator | Method |
|-----------|--------|
| HR | FFT on BVP signal → peak in 0.75-3.0 Hz range |
| HRV | NeuroKit2 peak detection → IBI intervals → SDNN, RMSSD, pNN50, LF/HF |
| Respiratory Rate | Hilbert transform → BVP envelope → FFT in 0.1-0.5 Hz range |
| Stress | Clinical heuristic based on RMSSD, SDNN, and LF/HF ratio |
| Emotion | FER (TensorFlow Lite) classifies facial expression at 1 frame/second |

### Reference algorithms

- **POS** — Wang et al., 2016 — Orthogonal projection of RGB channels
- **CHROM** — De Haan et al., 2013 — Chrominance-based model
- **NeuroKit2** — Makowski et al., 2021 — Physiological signal processing
- **FER** — Facial emotion recognition via TensorFlow Lite

---

## 💡 Tips for best results

| Tip | Why |
|-----|-----|
| ☀️ Good lighting (natural light) | More light = stronger signal = less noise |
| 🧍 Stay still | Movement creates artifacts in the signal |
| 📏 ~50cm distance | Face should fill a good portion of the frame |
| ⏱️ Minimum 30 seconds | More time = more cardiac cycles = better accuracy |
| 📱 Record on cellphone | Cellphone camera > streaming > webcam |
| 🚫 Avoid sunglasses | Reduces visible skin area |

---

## 📁 Project structure

```
facevitals/
├── mvp_rppg.py              # MVP v1 (HR only, 3 methods)
├── mvp_rppg_v2.py           # MVP v2 (HR + HRV + Resp + Stress + Emotion)
├── requirements_cpu.txt      # CPU dependencies
├── dashboard_saude.png       # Last generated dashboard
├── dashboard_saude.csv       # Measurement history
└── unsupervised_methods/     # POS, CHROM, GREEN algorithms
```

> **Note on language in source code:** Print statements, dashboard labels, and user-facing strings in the Python files are written in Portuguese (pt-BR). Code comments and docstrings are mixed. This is by design for the Brazilian end-user experience.

---

## 🚀 Roadmap — GPU evolution

With a GPU (NVIDIA T4 or RTX 3060+), the project can evolve to include:

| Feature | Status | Requires |
|---------|:------:|----------|
| Heart Rate (HR) | ✅ Implemented | CPU |
| HRV (SDNN, RMSSD) | ✅ Implemented | CPU |
| Respiratory Rate | ✅ Implemented | CPU |
| Stress Level | ✅ Implemented | CPU |
| Dominant Emotion | ✅ Implemented | CPU |
| Wellbeing Score | ✅ Implemented | CPU |
| Blood Pressure | 🔜 Planned | GPU + BP4D+ dataset |
| SpO2 (Oxygen Saturation) | 🔜 Planned | GPU |
| HR ±1-2 bpm (neural) | 🔜 Planned | GPU |
| High-fidelity HRV | 🔜 Planned | GPU |
| YOLO5Face detection | 🔜 Planned | GPU |

---

## ⚠️ Limitations

- 🏥 **Experimental data** — does not replace professional medical evaluations
- 💡 Poor lighting significantly degrades results
- 😐 Facial emotion is complementary (neutral expressions are often classified as "sad")
- ⏱️ Minimum 30 seconds of video for reliable results
- 🩸 Blood pressure and SpO2 require GPU + trained models

---

## 🏢 About

This project is under **internal testing at [Namu](https://namu.com.br)** — a Brazilian health and wellness platform.

- 🔓 **Open source** — no API Keys, no accounts, no external service dependencies
- 🇧🇷 **Output in Portuguese (pt-BR)** — dashboard, terminal, and CSV are in Brazilian Portuguese
- 💻 **Runs on any computer** — no GPU needed
- 📦 **Self-contained** — everything runs locally on your machine

---

## 📚 References

- [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) — Liu et al., 2022
- [POS - Algorithmic Principles of Remote PPG](https://ieeexplore.ieee.org/document/7565547) — Wang et al., 2016
- [CHROM - Robust Pulse Rate from Chrominance-based rPPG](https://ieeexplore.ieee.org/document/6523142) — De Haan et al., 2013
- [NeuroKit2](https://github.com/neuropsychology/NeuroKit) — Makowski et al., 2021
- [FER - Facial Expression Recognition](https://github.com/justinshenk/fer)

---

<p align="center">
  Made with 🫀 by <a href="https://namu.com.br">Namu</a>
</p>
