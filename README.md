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
  Extracts heart rate, HRV, respiratory rate, stress level and facial emotion from video<br>
  using remote photoplethysmography (rPPG). No GPU required.
</p>

<p align="center">
  <em>🇧🇷 Output em português — projeto em testes internos na <a href="https://namu.com.br">Namu</a></em>
</p>

---

## 🧐 O que é?

FaceVitals extrai dados de saúde analisando micro-variações de cor na pele do rosto capturadas por uma câmera comum. A cada batimento cardíaco, o fluxo sanguíneo altera microscopicamente a cor da pele — invisível a olho nu, mas detectável pela câmera.

**Sem sensores. Sem contato. Sem GPU. Sem API Key.**

```
Câmera → Frames → Detecção facial → ROI 72x72
                                        ↓
                              Média RGB por frame
                                        ↓
                              POS / CHROM (sinal BVP)
                                        ↓
                    ┌───────────┬────────────┬──────────────┐
                    FFT→HR    Picos→HRV    Envelope→Resp   FER→Emoção
                    └───────────┴────────────┴──────────────┘
                                        ↓
                              Score de Bem-Estar
                                        ↓
                           Dashboard PNG + CSV
```

---

## 📊 O que extrai

| Indicador | Descrição | Precisão (CPU) |
|-----------|-----------|:--------------:|
| 💓 Frequência Cardíaca (HR) | Batimentos por minuto | ±0-5 bpm |
| 📈 HRV (SDNN / RMSSD) | Variabilidade cardíaca | Moderada |
| 🌬️ Frequência Respiratória | Respirações por minuto | Estimativa |
| 🧠 Nível de Estresse (1-5) | Baseado em HRV (RMSSD, SDNN, LF/HF) | Moderada |
| 😐 Emoção Dominante | Expressão facial predominante | Complementar |
| ⭐ Score de Bem-Estar (0-100) | Combinação dos indicadores | Indicativo |

---

## ✅ Resultados validados

Teste com vídeo gravado no celular (boa iluminação), comparado com **VitalScan (Namu)** e **smartwatch Samsung**:

| Indicador | FaceVitals | VitalScan | Smartwatch |
|-----------|:----------:|:---------:|:----------:|
| HR | **81 bpm** | 81 bpm | 84 bpm |
| Estresse | **2 (Baixo)** | 2 | — |
| HRV (SDNN) | 61 ms | 94 ms | — |

> HR bateu **exatamente** com o VitalScan e estresse também. HRV tem margem de melhoria com GPU.

---

## ⚡ Início rápido

### Pré-requisitos

- Python 3.11+
- Webcam ou câmera de celular
- **Não precisa de GPU**
- **Não precisa de API Key**
- **Não precisa de conta em nenhum serviço**

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/facevitals.git
cd facevitals

# Crie o ambiente virtual
python -m venv venv

# Ative (Windows)
venv\Scripts\activate

# Ative (Linux/Mac)
source venv/bin/activate

# Atualize pip
python -m pip install --upgrade pip setuptools wheel

# Instale PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instale as dependências
pip install -r requirements_cpu.txt

# Instale o FER (detecção de emoções)
pip install fer
```

---

## 🎬 Como usar

### Input

FaceVitals aceita **3 tipos de entrada**:

#### 1. 📹 Vídeo gravado no celular (recomendado)

A melhor qualidade. Grave um vídeo selfie de 30+ segundos com boa iluminação e transfira para o computador.

```bash
python mvp_rppg_v2.py --video meu_video.mp4
```

#### 2. 💻 Webcam do notebook

Abre a câmera, mostra o preview com detecção facial em tempo real e captura por 30 segundos (ou o tempo que definir).

```bash
python mvp_rppg_v2.py --duration 30
```

#### 3. 📱 Câmera do celular como webcam

Usando apps como DroidCam ou Iriun Webcam. Selecione a câmera pelo índice.

```bash
# Liste as câmeras disponíveis e use o índice correto
python mvp_rppg_v2.py --camera 1 --duration 30
```

> **Nota:** vídeo gravado direto no celular > DroidCam > webcam (em qualidade de resultado)

### Parâmetros

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `--video` | Caminho do arquivo de vídeo | — |
| `--camera` | Índice da câmera (0, 1, 2...) | 0 |
| `--duration` | Duração da captura em segundos | 30 |
| `--output` | Nome do arquivo de saída | dashboard_saude.png |

---

## 📤 Output

Todos os resultados são gerados **em português**.

### 1. Terminal — Resumo em tempo real

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

### 2. Dashboard visual — `dashboard_saude.png`

Gráfico completo com:
- Painel de resumo dos indicadores
- Sinal BVP (pulso sanguíneo) com batimentos marcados
- Espectro de frequência cardíaca
- Sinal respiratório
- Intervalos entre batimentos (IBI)
- Indicadores de estresse e emoção
- Score de bem-estar

### 3. Dados exportados — `dashboard_saude.csv`

CSV com histórico de todas as medições, ideal para acompanhamento ao longo do tempo. Cada execução adiciona uma nova linha.

| Coluna | Exemplo |
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

## 🔬 Como funciona (técnico)

### Extração de sinais vitais

1. **Detecção facial** — Haar Cascade (OpenCV) localiza o rosto em cada frame
2. **ROI** — Recorta e redimensiona para 72x72 pixels
3. **Sinal RGB** — Calcula a média dos canais R, G, B de cada frame
4. **Algoritmos rPPG** — POS e CHROM processam o sinal RGB para isolar o pulso sanguíneo (BVP)
5. **Seleção automática** — Compara SNR dos dois métodos e usa o melhor

### Indicadores derivados

| Indicador | Método |
|-----------|--------|
| HR | FFT no sinal BVP → pico na faixa 0.75-3.0 Hz |
| HRV | NeuroKit2 detecta picos → intervalos IBI → SDNN, RMSSD, pNN50, LF/HF |
| Respiração | Transformada de Hilbert → envelope do BVP → FFT na faixa 0.1-0.5 Hz |
| Estresse | Heurística clínica baseada em RMSSD, SDNN e razão LF/HF |
| Emoção | FER (TensorFlow Lite) classifica expressão facial em 1 frame/segundo |

### Algoritmos de referência

- **POS** — Wang et al., 2016 — Projeção ortogonal dos canais RGB
- **CHROM** — De Haan et al., 2013 — Modelo de crominância
- **NeuroKit2** — Makowski et al., 2021 — Processamento de sinais fisiológicos
- **FER** — Detecção de emoções faciais via TensorFlow Lite

---

## 💡 Dicas para melhores resultados

| Dica | Por quê |
|------|---------|
| ☀️ Boa iluminação (luz natural) | Mais luz = mais sinal = menos ruído |
| 🧍 Fique parado | Movimento gera artefatos no sinal |
| 📏 ~50cm de distância | Rosto precisa ocupar boa parte do frame |
| ⏱️ Mínimo 30 segundos | Mais tempo = mais ciclos cardíacos = mais precisão |
| 📱 Grave no celular | Câmera do celular > streaming > webcam |
| 🚫 Evite óculos escuros | Reduz a área de pele visível |

---

## 📁 Estrutura do projeto

```
facevitals/
├── mvp_rppg.py              # MVP v1 (só HR, 3 métodos)
├── mvp_rppg_v2.py           # MVP v2 (HR + HRV + Resp + Estresse + Emoção)
├── requirements_cpu.txt      # Dependências para CPU
├── dashboard_saude.png       # Último dashboard gerado
├── dashboard_saude.csv       # Histórico de medições
├── configs/                  # Configurações YAML
├── unsupervised_methods/     # Algoritmos POS, CHROM, GREEN, etc.
├── neural_methods/           # Modelos neurais (requer GPU)
├── dataset/                  # Data loaders
└── evaluation/               # Métricas e pós-processamento
```

---

## 🚀 Roadmap — Evolução com GPU

Com uma GPU (NVIDIA T4 ou RTX 3060+), o projeto pode evoluir para:

| Funcionalidade | Status | Requer |
|---------------|:------:|--------|
| Frequência Cardíaca | ✅ Implementado | CPU |
| HRV (SDNN, RMSSD) | ✅ Implementado | CPU |
| Frequência Respiratória | ✅ Implementado | CPU |
| Nível de Estresse | ✅ Implementado | CPU |
| Emoção Dominante | ✅ Implementado | CPU |
| Score de Bem-Estar | ✅ Implementado | CPU |
| Pressão Arterial | 🔜 Planejado | GPU + BP4D+ |
| SpO2 (Saturação O₂) | 🔜 Planejado | GPU |
| HR ±1-2 bpm (neural) | 🔜 Planejado | GPU |
| HRV de alta fidelidade | 🔜 Planejado | GPU |
| Detecção facial YOLO5Face | 🔜 Planejado | GPU |

---

## ⚠️ Limitações

- 🏥 **Dados experimentais** — não substituem avaliações médicas profissionais
- 💡 Iluminação ruim degrada significativamente os resultados
- 😐 Emoção facial é complementar (expressões neutras podem ser classificadas como "triste")
- ⏱️ Mínimo 30 segundos de vídeo para resultados confiáveis
- 🩸 Pressão arterial e SpO2 requerem GPU + modelos treinados

---

## 🏢 Sobre

Projeto em **testes internos na [Namu](https://namu.com.br)** — plataforma brasileira de saúde e bem-estar.

- 🔓 **Open source** — sem API Keys, sem contas, sem dependências externas de serviços
- 🇧🇷 **Output em português** — dashboard, terminal e CSV em pt-BR
- 💻 **Roda em qualquer computador** — sem necessidade de GPU
- 📦 **Self-contained** — tudo roda localmente na sua máquina

---

## 📚 Referências

- [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) — Liu et al., 2022
- [POS - Algorithmic Principles of Remote PPG](https://ieeexplore.ieee.org/document/7565547) — Wang et al., 2016
- [CHROM - Robust Pulse Rate from Chrominance-based rPPG](https://ieeexplore.ieee.org/document/6523142) — De Haan et al., 2013
- [NeuroKit2](https://github.com/neuropsychology/NeuroKit) — Makowski et al., 2021
- [FER - Facial Expression Recognition](https://github.com/justinshenk/fer)

---

<p align="center">
  Feito com 🫀 por <a href="https://namu.com.br">Namu</a>
</p>
