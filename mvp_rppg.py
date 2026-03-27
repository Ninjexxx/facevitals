"""
MVP - Extração de Frequência Cardíaca via rPPG (Remote Photoplethysmography)
Funciona com webcam ao vivo ou arquivo de vídeo.
Usa algoritmos não-supervisionados do rPPG-Toolbox (POS, CHROM, GREEN).
Não requer GPU.

Uso:
    python mvp_rppg.py                    # Webcam ao vivo (grava 30s)
    python mvp_rppg.py --video meu.mp4    # Arquivo de vídeo
    python mvp_rppg.py --duration 20      # Webcam por 20 segundos
"""

import sys
import os
import argparse

import cv2
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Adiciona o diretório raiz do toolbox ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unsupervised_methods.methods.POS_WANG import POS_WANG
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from unsupervised_methods.methods.GREEN import GREEN


def detect_face(frame, face_cascade):
    """Detecta rosto e retorna a região recortada."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # Aumenta a bounding box em 50%
    pad_w, pad_h = int(w * 0.25), int(h * 0.25)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame.shape[1], x + w + pad_w)
    y2 = min(frame.shape[0], y + h + pad_h)
    face = frame[y1:y2, x1:x2]
    return cv2.resize(face, (72, 72)), (x1, y1, x2, y2)


def capture_frames(source, duration, target_fps=30):
    """Captura frames de webcam ou vídeo."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frames_rgb = []
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    is_webcam = isinstance(source, int)
    max_frames = int(duration * fps) if is_webcam else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames <= 0:
        max_frames = int(duration * fps)

    print(f"Capturando frames (FPS: {fps:.0f})...")
    if is_webcam:
        print(f"Olhe para a câmera por {duration}s. Pressione 'q' para parar.")

    frame_count = 0
    last_bbox = None
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        face_roi, bbox = detect_face(frame, face_cascade)
        if face_roi is not None:
            last_bbox = bbox
            frames_rgb.append(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        elif last_bbox is not None:
            # Usa última bbox conhecida
            x1, y1, x2, y2 = last_bbox
            face_roi = frame[y1:y2, x1:x2]
            frames_rgb.append(cv2.cvtColor(cv2.resize(face_roi, (72, 72)), cv2.COLOR_BGR2RGB))

        if is_webcam:
            display = frame.copy()
            if last_bbox:
                x1, y1, x2, y2 = last_bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elapsed = frame_count / fps
            cv2.putText(display, f"Gravando: {elapsed:.1f}s / {duration}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("rPPG - Captura", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    if is_webcam:
        cv2.destroyAllWindows()

    print(f"Frames capturados com rosto: {len(frames_rgb)}")
    return np.array(frames_rgb), fps


def estimate_hr_from_bvp(bvp, fs):
    """Estima HR (bpm) a partir do sinal BVP usando FFT."""
    if len(bvp) < 10:
        return 0.0, np.array([]), np.array([])
    # FFT
    N = len(bvp)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(bvp))
    # Filtra na faixa de HR normal: 0.75 Hz (45 bpm) a 3.0 Hz (180 bpm)
    mask = (freqs >= 0.75) & (freqs <= 3.0)
    if not np.any(mask):
        return 0.0, freqs, fft_mag
    freqs_filtered = freqs[mask]
    fft_filtered = fft_mag[mask]
    peak_freq = freqs_filtered[np.argmax(fft_filtered)]
    hr_bpm = peak_freq * 60.0
    return hr_bpm, freqs, fft_mag


def run_analysis(frames, fps):
    """Executa os 3 métodos e retorna resultados."""
    results = {}
    methods = {
        "POS": lambda f: POS_WANG(f, fps),
        "CHROM": lambda f: CHROME_DEHAAN(f, fps),
        "GREEN": lambda f: GREEN(f),
    }

    for name, method in methods.items():
        print(f"  Processando método {name}...")
        try:
            bvp = method(frames)
            bvp = np.squeeze(np.array(bvp))
            hr, freqs, fft_mag = estimate_hr_from_bvp(bvp, fps)
            results[name] = {"bvp": bvp, "hr": hr, "freqs": freqs, "fft_mag": fft_mag}
        except Exception as e:
            print(f"    Erro no método {name}: {e}")
            results[name] = {"bvp": np.array([]), "hr": 0.0, "freqs": np.array([]), "fft_mag": np.array([])}

    return results


def plot_results(results, fps, output_path="resultado_rppg.png"):
    """Gera gráfico com sinais BVP e espectro de frequência."""
    n_methods = len(results)
    fig, axes = plt.subplots(n_methods, 2, figsize=(14, 4 * n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)

    for i, (name, data) in enumerate(results.items()):
        bvp = data["bvp"]
        hr = data["hr"]

        # Sinal BVP no tempo
        if len(bvp) > 0:
            t = np.arange(len(bvp)) / fps
            axes[i, 0].plot(t, bvp, 'b-', linewidth=0.5)
            axes[i, 0].set_title(f"{name} - Sinal BVP (HR estimado: {hr:.1f} bpm)")
        else:
            axes[i, 0].set_title(f"{name} - Sem dados")
        axes[i, 0].set_xlabel("Tempo (s)")
        axes[i, 0].set_ylabel("Amplitude")

        # Espectro FFT
        freqs, fft_mag = data["freqs"], data["fft_mag"]
        if len(freqs) > 0:
            mask = (freqs >= 0.5) & (freqs <= 3.5)
            axes[i, 1].plot(freqs[mask] * 60, fft_mag[mask], 'r-')
            axes[i, 1].axvline(x=hr, color='g', linestyle='--', label=f'HR={hr:.1f} bpm')
            axes[i, 1].legend()
            axes[i, 1].set_title(f"{name} - Espectro de Frequência")
        else:
            axes[i, 1].set_title(f"{name} - Sem dados")
        axes[i, 1].set_xlabel("Frequência Cardíaca (bpm)")
        axes[i, 1].set_ylabel("Magnitude")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nGráfico salvo em: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="MVP rPPG - Extração de Frequência Cardíaca")
    parser.add_argument("--video", type=str, default=None, help="Caminho do vídeo (omita para webcam)")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera (0=padrão, 1, 2...)")
    parser.add_argument("--duration", type=int, default=30, help="Duração da captura em segundos (webcam)")
    parser.add_argument("--output", type=str, default="resultado_rppg.png", help="Arquivo de saída do gráfico")
    args = parser.parse_args()

    source = args.video if args.video else args.camera
    print("=" * 60)
    print("  MVP rPPG - Extração de Frequência Cardíaca pela Pele")
    print("=" * 60)

    # 1. Captura
    print("\n[1/3] Capturando vídeo...")
    frames, fps = capture_frames(source, args.duration)
    if len(frames) < 30:
        print("ERRO: Poucos frames capturados. Verifique a câmera/vídeo e iluminação.")
        return

    # 2. Análise
    print(f"\n[2/3] Analisando {len(frames)} frames ({len(frames)/fps:.1f}s de vídeo)...")
    results = run_analysis(frames, fps)

    # 3. Resultados
    print("\n[3/3] Resultados:")
    print("-" * 40)
    hrs = []
    for name, data in results.items():
        hr = data["hr"]
        print(f"  {name:8s}: {hr:6.1f} bpm")
        if hr > 0:
            hrs.append(hr)

    if hrs:
        avg_hr = np.mean(hrs)
        print("-" * 40)
        print(f"  {'MÉDIA':8s}: {avg_hr:6.1f} bpm")
        print("-" * 40)

        if 50 <= avg_hr <= 100:
            print("  Status: Frequência cardíaca dentro da faixa normal de repouso.")
        elif 40 <= avg_hr < 50:
            print("  Status: Bradicardia leve (pode ser normal em atletas).")
        elif 100 < avg_hr <= 120:
            print("  Status: Taquicardia leve.")
        else:
            print("  Status: Valor fora da faixa típica - verifique condições de captura.")

    plot_results(results, fps, args.output)
    print("\nDica: Para melhores resultados, garanta boa iluminação e fique parado.")


if __name__ == "__main__":
    main()
