"""
MVP v2 - Extração de Dados de Saúde via rPPG
Extrai: HR, HRV, Frequência Respiratória e Nível de Estresse.

Uso:
    python mvp_rppg_v2.py --video meu_video.mp4
    python mvp_rppg_v2.py --duration 30
    python mvp_rppg_v2.py --camera 1 --duration 30
"""

import sys
import os
import argparse
import json
from datetime import datetime

import cv2
import numpy as np
from scipy import signal as scipy_signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neurokit2 as nk
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unsupervised_methods.methods.POS_WANG import POS_WANG
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from fer.fer import FER


def detect_face(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_w, pad_h = int(w * 0.25), int(h * 0.25)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame.shape[1], x + w + pad_w)
    y2 = min(frame.shape[0], y + h + pad_h)
    face = frame[y1:y2, x1:x2]
    return cv2.resize(face, (72, 72)), (x1, y1, x2, y2)


def capture_frames(source, duration, target_fps=30):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir: {source}")

    nominal_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frames_rgb = []
    raw_faces_bgr = []
    timestamps = []
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    is_webcam = isinstance(source, int)
    max_frames = int(duration * nominal_fps) if is_webcam else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames <= 0:
        max_frames = int(duration * nominal_fps)

    print(f"  Capturando frames (FPS nominal: {nominal_fps:.0f})...")
    if is_webcam:
        print(f"  Olhe para a camera por {duration}s. Pressione 'q' para parar.")

    import time
    t_start = time.perf_counter()
    frame_count = 0
    last_bbox = None
    sample_interval = max(1, int(nominal_fps))
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        timestamps.append(time.perf_counter() - t_start)

        face_roi, bbox = detect_face(frame, face_cascade)
        if face_roi is not None:
            last_bbox = bbox
            frames_rgb.append(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        elif last_bbox is not None:
            x1, y1, x2, y2 = last_bbox
            face_roi = cv2.resize(frame[y1:y2, x1:x2], (72, 72))
            frames_rgb.append(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

        if frame_count % sample_interval == 0:
            if last_bbox:
                x1, y1, x2, y2 = last_bbox
                raw_faces_bgr.append(frame[y1:y2, x1:x2].copy())
            else:
                raw_faces_bgr.append(frame)

        if is_webcam:
            if last_bbox:
                x1, y1, x2, y2 = last_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elapsed = frame_count / nominal_fps
            cv2.putText(frame, f"Gravando: {elapsed:.1f}s / {duration}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("rPPG - Captura", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    if is_webcam:
        cv2.destroyAllWindows()

    # Calcula FPS real e jitter para webcam
    if is_webcam and len(timestamps) > 1:
        total_time = timestamps[-1] - timestamps[0]
        real_fps = (len(timestamps) - 1) / total_time if total_time > 0 else nominal_fps
        intervals = np.diff(timestamps)
        jitter_ms = np.std(intervals) * 1000
        print(f"  FPS real: {real_fps:.1f} (nominal: {nominal_fps:.0f}, jitter: {jitter_ms:.1f}ms)")
        if jitter_ms > 30:
            print(f"  AVISO: Jitter alto ({jitter_ms:.1f}ms) - resultados podem ser menos precisos")
        fps = real_fps
    else:
        fps = nominal_fps

    print(f"  Frames com rosto: {len(frames_rgb)}")
    print(f"  Frames para emocao: {len(raw_faces_bgr)}")
    return np.array(frames_rgb), fps, raw_faces_bgr


# Faixa de frequência cardíaca: 0.7 Hz (42 bpm) a 4.0 Hz (240 bpm)
HR_FREQ_LOW = 0.7
HR_FREQ_HIGH = 4.0
SNR_MIN_THRESHOLD = -15.0  # dB - abaixo disso o sinal é considerado instável


def calc_snr(bvp, fs):
    """Calcula SNR do sinal BVP na faixa cardíaca."""
    freqs = np.fft.rfftfreq(len(bvp), d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(bvp))
    mask = (freqs >= HR_FREQ_LOW) & (freqs <= HR_FREQ_HIGH)
    if not np.any(mask):
        return -np.inf
    peak_power = np.max(fft_mag[mask]) ** 2
    total_power = np.sum(fft_mag[mask] ** 2)
    noise_power = total_power - peak_power
    if noise_power <= 0:
        return 0
    return 10 * np.log10(peak_power / noise_power)


def extract_bvp(frames, fps):
    """Extrai BVP usando POS e CHROM, retorna o melhor sinal."""
    bvp_pos = np.squeeze(POS_WANG(frames, fps))
    bvp_chrom = np.squeeze(CHROME_DEHAAN(frames, fps))

    snr_pos = calc_snr(bvp_pos, fps)
    snr_chrom = calc_snr(bvp_chrom, fps)
    best_snr = max(snr_pos, snr_chrom)

    # Alerta de sinal instável
    signal_quality = "OK"
    if best_snr < SNR_MIN_THRESHOLD:
        signal_quality = "INSTAVEL"
        print(f"  AVISO: Sinal instavel (SNR: {best_snr:.1f} dB < {SNR_MIN_THRESHOLD} dB)")
        print(f"  Possivel causa: movimento excessivo ou iluminacao insuficiente")

    if snr_pos >= snr_chrom:
        print(f"  Metodo selecionado: POS (SNR: {snr_pos:.1f} dB)")
        return bvp_pos, "POS", signal_quality
    else:
        print(f"  Metodo selecionado: CHROM (SNR: {snr_chrom:.1f} dB)")
        return bvp_chrom, "CHROM", signal_quality


def estimate_hr(bvp, fps):
    """Estima HR via FFT com filtro de banda estrita [0.7-4.0 Hz]."""
    # Aplica filtro passa-faixa estrito antes da FFT
    nyq = fps / 2
    low = HR_FREQ_LOW / nyq
    high = min(HR_FREQ_HIGH / nyq, 0.99)
    b, a = scipy_signal.butter(3, [low, high], btype='bandpass')
    bvp_filtered = scipy_signal.filtfilt(b, a, bvp.astype(np.double))

    freqs = np.fft.rfftfreq(len(bvp_filtered), d=1.0 / fps)
    fft_mag = np.abs(np.fft.rfft(bvp_filtered))
    mask = (freqs >= HR_FREQ_LOW) & (freqs <= HR_FREQ_HIGH)
    if not np.any(mask):
        return 0.0, freqs, fft_mag
    peak_freq = freqs[mask][np.argmax(fft_mag[mask])]
    return peak_freq * 60.0, freqs, fft_mag


def analyze_hrv(bvp, fps, hr_bpm):
    """Analisa HRV (variabilidade da frequência cardíaca)."""
    # Filtra o BVP mais agressivamente ao redor do HR detectado
    hr_hz = hr_bpm / 60.0
    nyq = fps / 2
    low = max(0.75, hr_hz - 0.5) / nyq
    high = min(3.0, hr_hz + 0.5) / nyq
    if high >= 1.0:
        high = 0.99
    b, a = scipy_signal.butter(3, [low, high], btype='bandpass')
    bvp_filtered = scipy_signal.filtfilt(b, a, bvp.astype(np.double))

    # Detecta picos no sinal filtrado
    bvp_clean = nk.ppg_clean(bvp_filtered, sampling_rate=int(fps))
    info = nk.ppg_findpeaks(bvp_clean, sampling_rate=int(fps))
    peaks = info["PPG_Peaks"]

    if len(peaks) < 4:
        return None

    # Intervalos entre picos (IBI) em milissegundos
    ibi_samples = np.diff(peaks)
    ibi_ms = (ibi_samples / fps) * 1000

    # Filtra IBIs baseado no HR detectado (±30% da média esperada)
    expected_ibi = 60000.0 / hr_bpm
    ibi_low = expected_ibi * 0.7
    ibi_high = expected_ibi * 1.3
    valid = (ibi_ms >= ibi_low) & (ibi_ms <= ibi_high)
    ibi_ms = ibi_ms[valid]

    if len(ibi_ms) < 3:
        return None

    # Remove outliers: descarta IBIs fora de 2 desvios padrão da mediana
    median_ibi = np.median(ibi_ms)
    mad = np.median(np.abs(ibi_ms - median_ibi))
    if mad > 0:
        valid2 = np.abs(ibi_ms - median_ibi) < 2 * mad * 1.4826  # MAD to std
        if np.sum(valid2) >= 3:
            ibi_ms = ibi_ms[valid2]

    # Métricas HRV no domínio do tempo
    sdnn = np.std(ibi_ms, ddof=1)   # Desvio padrão dos intervalos NN
    rmssd = np.sqrt(np.mean(np.diff(ibi_ms) ** 2))  # RMSSD
    mean_ibi = np.mean(ibi_ms)
    nn_diffs = np.abs(np.diff(ibi_ms))
    pnn50 = np.sum(nn_diffs > 50) / len(nn_diffs) * 100 if len(nn_diffs) > 0 else 0

    # Análise espectral do HRV (domínio da frequência)
    lf_power, hf_power, lf_hf_ratio = compute_hrv_frequency(ibi_ms)

    return {
        "mean_ibi_ms": round(mean_ibi, 1),
        "sdnn_ms": round(sdnn, 1),
        "rmssd_ms": round(rmssd, 1),
        "pnn50_pct": round(pnn50, 1),
        "lf_power": round(lf_power, 2),
        "hf_power": round(hf_power, 2),
        "lf_hf_ratio": round(lf_hf_ratio, 2),
        "peaks": peaks,
        "ibi_ms": ibi_ms,
    }


def compute_hrv_frequency(ibi_ms):
    """Calcula potência LF, HF e razão LF/HF do HRV."""
    if len(ibi_ms) < 8:
        return 0, 0, 0

    # Interpola IBI para amostragem uniforme
    ibi_s = ibi_ms / 1000.0
    cumulative_time = np.cumsum(ibi_s)
    cumulative_time = cumulative_time - cumulative_time[0]

    # Reamostra a 4 Hz
    resample_rate = 4.0
    t_uniform = np.arange(0, cumulative_time[-1], 1.0 / resample_rate)
    ibi_interp = np.interp(t_uniform, cumulative_time, ibi_s)
    ibi_interp = ibi_interp - np.mean(ibi_interp)

    # PSD via Welch
    freqs, psd = scipy_signal.welch(ibi_interp, fs=resample_rate, nperseg=min(len(ibi_interp), 256))

    # Bandas: LF (0.04-0.15 Hz), HF (0.15-0.4 Hz)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.4)

    lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
    hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0

    return lf_power, hf_power, lf_hf_ratio


def estimate_respiratory_rate(bvp, fps):
    """Estima frequência respiratória pela modulação de amplitude do BVP."""
    # Envelope do sinal (modulação respiratória)
    analytic = scipy_signal.hilbert(bvp)
    envelope = np.abs(analytic)

    # Filtra na faixa respiratória: 0.1-0.5 Hz (6-30 respirações/min)
    nyq = fps / 2
    low, high = 0.1 / nyq, 0.5 / nyq
    if high >= 1.0:
        high = 0.99
    b, a = scipy_signal.butter(2, [low, high], btype='bandpass')
    resp_signal = scipy_signal.filtfilt(b, a, envelope)

    # FFT para encontrar pico respiratório
    freqs = np.fft.rfftfreq(len(resp_signal), d=1.0 / fps)
    fft_mag = np.abs(np.fft.rfft(resp_signal))
    mask = (freqs >= 0.1) & (freqs <= 0.5)

    if not np.any(mask):
        return 0.0, resp_signal

    peak_freq = freqs[mask][np.argmax(fft_mag[mask])]
    rr_bpm = peak_freq * 60.0

    return rr_bpm, resp_signal


def analyze_emotions(raw_frames_bgr):
    """Detecta emoções faciais nos frames amostrados."""
    detector = FER(mtcnn=False)

    EMOTION_MAP = {
        'angry': 'Raiva',
        'disgust': 'Nojo',
        'fear': 'Medo',
        'happy': 'Feliz',
        'sad': 'Triste',
        'surprise': 'Surpresa',
        'neutral': 'Neutro',
    }

    all_scores = {e: [] for e in EMOTION_MAP}
    dominant_emotions = []

    for frame in raw_frames_bgr:
        result = detector.detect_emotions(frame)
        if result:
            emotions = result[0]['emotions']
            for e, score in emotions.items():
                if e in all_scores:
                    all_scores[e].append(score)
            dominant = max(emotions, key=emotions.get)
            dominant_emotions.append(dominant)

    if not dominant_emotions:
        return None

    # Emoção dominante = a mais frequente nos frames amostrados
    from collections import Counter
    most_common = Counter(dominant_emotions).most_common(1)[0][0]

    return {
        'dominant': most_common,
        'dominant_pt': EMOTION_MAP[most_common],
    }


def compute_wellbeing_score(hr, hrv_data, rr, stress_label, emotion_data):
    """Calcula score de bem-estar geral (0-100)."""
    score = 50  # Base

    # HR: 60-80 ideal
    if 60 <= hr <= 80:
        score += 10
    elif 50 <= hr <= 90:
        score += 5
    else:
        score -= 5

    # HRV
    if hrv_data:
        sdnn = hrv_data['sdnn_ms']
        if 50 <= sdnn <= 120:
            score += 10
        elif sdnn > 120:
            score += 5
        else:
            score -= 5

    # Respiração: 12-20 ideal
    if 12 <= rr <= 20:
        score += 10
    elif 8 <= rr <= 24:
        score += 5
    else:
        score -= 5

    # Estresse
    if '1' in stress_label or '2' in stress_label:
        score += 10
    elif '4' in stress_label or '5' in stress_label:
        score -= 10

    return max(0, min(100, score))


def assess_stress(hrv_data):
    """Avalia nível de estresse baseado no HRV."""
    if hrv_data is None:
        return "Dados insuficientes", "gray"

    lf_hf = hrv_data["lf_hf_ratio"]
    rmssd = hrv_data["rmssd_ms"]
    sdnn = hrv_data["sdnn_ms"]

    # Escala 1-5 baseada em literatura clínica
    # RMSSD: <20ms=estresse alto, 20-50ms=moderado, >50ms=relaxado
    # SDNN: <50ms=estresse alto, 50-100ms=normal, >100ms=relaxado
    # LF/HF: >2.5=estresse, 1-2.5=normal, <1=relaxado
    score = 3  # Começa no meio (normal)

    if rmssd < 20:
        score += 1
    elif rmssd > 50:
        score -= 1

    if sdnn < 50:
        score += 1
    elif sdnn > 100:
        score -= 1

    if lf_hf > 2.5:
        score += 1
    elif lf_hf < 1.0:
        score -= 1

    score = max(1, min(5, score))

    labels = {
        1: ("1 - Muito baixo (relaxado)", "#2ecc71"),
        2: ("2 - Baixo", "#27ae60"),
        3: ("3 - Moderado (normal)", "#f39c12"),
        4: ("4 - Elevado", "#e67e22"),
        5: ("5 - Alto", "#e74c3c"),
    }
    return labels[score]


def plot_health_dashboard(bvp, fps, hr, hrv_data, rr, resp_signal, stress_level, stress_color, method_name, emotion_data, wellbeing, output_path):
    """Gera dashboard visual com todos os dados de saúde."""
    fig = plt.figure(figsize=(16, 18))
    fig.suptitle("Dashboard de Saude - rPPG + Emotion Analysis", fontsize=16, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(5, 2, hspace=0.45, wspace=0.3, top=0.93, bottom=0.04)

    # --- Painel de resumo ---
    ax_summary = fig.add_subplot(gs[0, :])
    ax_summary.axis('off')

    stress_label, _ = stress_level, stress_color
    summary_text = (
        f"Frequencia Cardiaca:  {hr:.0f} bpm     |     "
        f"Freq. Respiratoria:  {rr:.0f} rpm     |     "
        f"Estresse:  {stress_label}"
    )
    ax_summary.text(0.5, 0.7, summary_text, transform=ax_summary.transAxes,
                    fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', edgecolor='#2c3e50'))

    if hrv_data:
        hrv_text = (
            f"HRV  ->  SDNN: {hrv_data['sdnn_ms']:.0f} ms  |  "
            f"RMSSD: {hrv_data['rmssd_ms']:.0f} ms  |  "
            f"pNN50: {hrv_data['pnn50_pct']:.0f}%  |  "
            f"LF/HF: {hrv_data['lf_hf_ratio']:.1f}"
        )
        ax_summary.text(0.5, 0.2, hrv_text, transform=ax_summary.transAxes,
                        fontsize=11, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='#d5f5e3', edgecolor='#27ae60'))

    # --- Sinal BVP ---
    ax_bvp = fig.add_subplot(gs[1, :])
    t = np.arange(len(bvp)) / fps
    ax_bvp.plot(t, bvp, '#2980b9', linewidth=0.6)
    if hrv_data and len(hrv_data["peaks"]) > 0:
        peak_times = hrv_data["peaks"] / fps
        valid_peaks = hrv_data["peaks"][hrv_data["peaks"] < len(bvp)]
        ax_bvp.plot(valid_peaks / fps, bvp[valid_peaks], 'rv', markersize=4, label='Batimentos')
        ax_bvp.legend(loc='upper right')
    ax_bvp.set_title(f'Sinal BVP - Pulso Sanguineo ({method_name})', fontweight='bold')
    ax_bvp.set_xlabel('Tempo (s)')
    ax_bvp.set_ylabel('Amplitude')

    # --- Espectro HR ---
    ax_fft = fig.add_subplot(gs[2, 0])
    freqs = np.fft.rfftfreq(len(bvp), d=1.0 / fps)
    fft_mag = np.abs(np.fft.rfft(bvp))
    mask = (freqs >= 0.5) & (freqs <= 3.5)
    ax_fft.plot(freqs[mask] * 60, fft_mag[mask], '#e74c3c')
    ax_fft.axvline(x=hr, color='green', linestyle='--', linewidth=2, label=f'HR = {hr:.0f} bpm')
    ax_fft.set_title('Espectro - Frequencia Cardiaca', fontweight='bold')
    ax_fft.set_xlabel('BPM')
    ax_fft.set_ylabel('Magnitude')
    ax_fft.legend()

    # --- Sinal respiratório ---
    ax_resp = fig.add_subplot(gs[2, 1])
    t_resp = np.arange(len(resp_signal)) / fps
    ax_resp.plot(t_resp, resp_signal, '#27ae60', linewidth=0.8)
    ax_resp.set_title(f'Sinal Respiratorio ({rr:.0f} rpm)', fontweight='bold')
    ax_resp.set_xlabel('Tempo (s)')
    ax_resp.set_ylabel('Amplitude')

    # --- IBI (intervalos entre batimentos) ---
    ax_ibi = fig.add_subplot(gs[3, 0])
    if hrv_data and len(hrv_data["ibi_ms"]) > 0:
        ax_ibi.plot(hrv_data["ibi_ms"], '#8e44ad', marker='o', markersize=3, linewidth=1)
        ax_ibi.axhline(y=hrv_data["mean_ibi_ms"], color='gray', linestyle='--',
                        label=f'Media: {hrv_data["mean_ibi_ms"]:.0f} ms')
        ax_ibi.legend()
    ax_ibi.set_title('Intervalos Entre Batimentos (IBI)', fontweight='bold')
    ax_ibi.set_xlabel('Batimento #')
    ax_ibi.set_ylabel('IBI (ms)')

    # --- Indicador de estresse ---
    ax_stress = fig.add_subplot(gs[3, 1])
    ax_stress.axis('off')
    ax_stress.text(0.5, 0.6, "Nivel de Estresse", transform=ax_stress.transAxes,
                   fontsize=14, ha='center', va='center', fontweight='bold')
    ax_stress.text(0.5, 0.35, stress_label, transform=ax_stress.transAxes,
                   fontsize=20, ha='center', va='center', color=stress_color, fontweight='bold')
    if hrv_data:
        detail = f"(RMSSD={hrv_data['rmssd_ms']:.0f}ms, LF/HF={hrv_data['lf_hf_ratio']:.1f})"
        ax_stress.text(0.5, 0.12, detail, transform=ax_stress.transAxes,
                       fontsize=10, ha='center', va='center', color='gray')

    # --- Emoção dominante + Score de bem-estar ---
    ax_bottom = fig.add_subplot(gs[4, 0])
    ax_bottom.axis('off')
    if emotion_data:
        ax_bottom.text(0.5, 0.6, 'Emocao Dominante', transform=ax_bottom.transAxes,
                       fontsize=14, ha='center', va='center', fontweight='bold')
        ax_bottom.text(0.5, 0.3, emotion_data['dominant_pt'], transform=ax_bottom.transAxes,
                       fontsize=22, ha='center', va='center', color='#8e44ad', fontweight='bold')
    else:
        ax_bottom.text(0.5, 0.5, 'Emocao: sem dados', ha='center', va='center',
                       transform=ax_bottom.transAxes)

    ax_well = fig.add_subplot(gs[4, 1])
    ax_well.axis('off')
    well_color = '#2ecc71' if wellbeing >= 70 else '#f39c12' if wellbeing >= 40 else '#e74c3c'
    ax_well.text(0.5, 0.6, 'Score de Bem-Estar', transform=ax_well.transAxes,
                 fontsize=14, ha='center', va='center', fontweight='bold')
    ax_well.text(0.5, 0.3, f'{wellbeing}/100', transform=ax_well.transAxes,
                 fontsize=32, ha='center', va='center', color=well_color, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Dashboard salvo em: {output_path}")
    plt.close()


def export_csv(hr, hrv_data, rr, stress_label, method_name, emotion_data, wellbeing, output_path):
    """Exporta resultados em CSV."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "method": method_name,
        "hr_bpm": round(hr, 1),
        "rr_rpm": round(rr, 1),
        "stress": stress_label,
    }
    if hrv_data:
        data.update({
            "sdnn_ms": hrv_data["sdnn_ms"],
            "rmssd_ms": hrv_data["rmssd_ms"],
            "pnn50_pct": hrv_data["pnn50_pct"],
            "lf_hf_ratio": hrv_data["lf_hf_ratio"],
            "mean_ibi_ms": hrv_data["mean_ibi_ms"],
        })
    if emotion_data:
        data["emotion"] = emotion_data["dominant_pt"]
    data["wellbeing_score"] = wellbeing

    csv_path = output_path.replace('.png', '.csv')
    header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if header:
            f.write(','.join(data.keys()) + '\n')
        f.write(','.join(str(v) for v in data.values()) + '\n')
    print(f"  Dados exportados em: {csv_path}")

    return data


def main():
    parser = argparse.ArgumentParser(description="MVP v2 - Dados de Saude via rPPG")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--output", type=str, default="dashboard_saude.png")
    args = parser.parse_args()

    source = args.video if args.video else args.camera

    print("=" * 60)
    print("  MVP v2 - Extracao de Dados de Saude via rPPG")
    print("=" * 60)

    # 1. Captura
    print("\n[1/5] Capturando video...")
    frames, fps, raw_frames = capture_frames(source, args.duration)
    if len(frames) < 60:
        print("ERRO: Poucos frames. Verifique camera/video e iluminacao.")
        return

    # 2. Extração BVP
    print(f"\n[2/5] Extraindo sinal de pulso ({len(frames)} frames, {len(frames)/fps:.1f}s)...")
    bvp, method_name, signal_quality = extract_bvp(frames, fps)

    # 3. Frequência cardíaca
    print("\n[3/5] Calculando frequencia cardiaca...")
    hr, _, _ = estimate_hr(bvp, fps)
    print(f"  HR: {hr:.0f} bpm")
    if signal_quality == "INSTAVEL":
        print(f"  *** SINAL INSTAVEL - valor pode nao ser confiavel ***")

    # 4. HRV
    print("\n[4/5] Analisando variabilidade cardiaca (HRV)...")
    hrv_data = analyze_hrv(bvp, fps, hr)
    if hrv_data:
        print(f"  SDNN:    {hrv_data['sdnn_ms']:.0f} ms (variabilidade geral)")
        print(f"  RMSSD:   {hrv_data['rmssd_ms']:.0f} ms (atividade parassimpatica)")
        print(f"  pNN50:   {hrv_data['pnn50_pct']:.0f}% (estabilidade)")
        print(f"  LF/HF:   {hrv_data['lf_hf_ratio']:.1f} (balanco autonomico)")
    else:
        print("  Dados insuficientes para HRV")

    # 5. Respiração + Estresse
    print("\n[5/5] Estimando respiracao e estresse...")
    rr, resp_signal = estimate_respiratory_rate(bvp, fps)
    stress_label, stress_color = assess_stress(hrv_data)
    print(f"  Freq. Respiratoria: {rr:.0f} rpm")
    print(f"  Nivel de Estresse:  {stress_label}")



    # 6. Emoções
    print("\n[6/6] Analisando emocoes faciais...")
    emotion_data = analyze_emotions(raw_frames)
    if emotion_data:
        print(f"  Emocao dominante: {emotion_data['dominant_pt']}")
    else:
        print("  Nao foi possivel detectar emocoes")

    # Score de bem-estar
    wellbeing = compute_wellbeing_score(hr, hrv_data, rr, stress_label, emotion_data)

    # Resumo final atualizado
    print("\n" + "=" * 60)
    print("  RESUMO DE SAUDE")
    print("=" * 60)
    print(f"  Frequencia Cardiaca:    {hr:.0f} bpm")
    if hrv_data:
        print(f"  Variabilidade (SDNN):   {hrv_data['sdnn_ms']:.0f} ms")
        print(f"  Variabilidade (RMSSD):  {hrv_data['rmssd_ms']:.0f} ms")
    print(f"  Freq. Respiratoria:     {rr:.0f} rpm")
    print(f"  Nivel de Estresse:      {stress_label}")
    if emotion_data:
        print(f"  Emocao Dominante:       {emotion_data['dominant_pt']}")
    print(f"  Score de Bem-Estar:     {wellbeing}/100")
    if signal_quality == "INSTAVEL":
        print(f"  *** ATENCAO: Sinal instavel - repita com melhor iluminacao/estabilidade ***")
    print("=" * 60)

    # Gera dashboard e exporta CSV
    plot_health_dashboard(bvp, fps, hr, hrv_data, rr, resp_signal,
                          stress_label, stress_color, method_name,
                          emotion_data, wellbeing, args.output)
    export_csv(hr, hrv_data, rr, stress_label, method_name, emotion_data, wellbeing, args.output)

    print("\n  Aviso: Estes dados sao experimentais e NAO substituem")
    print("  avaliacoes medicas profissionais.")


if __name__ == "__main__":
    main()
