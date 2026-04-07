"""
Pre-validacao de qualidade do frame usando Moondream2 (VLM).
Analisa condicoes de captura antes do processamento rPPG.

Verifica: iluminacao, oculos/mascara, rosto visivel, distancia.
Retorna avisos e score de qualidade (0-100).

Usa transformers (HuggingFace) para rodar 100% local, sem API key.
"""

import cv2

_model = None
_tokenizer = None

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26"


def _load_model():
    """Carrega Moondream2 via transformers (local, CPU, float16 para economizar RAM)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print("  Baixando/carregando modelo (~1.8GB na primeira vez)...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        _model.eval()
        return _model, _tokenizer
    except Exception as e:
        print(f"  AVISO: Erro ao carregar Moondream: {e}")
        return None, None


def _grab_first_frame(source):
    """Captura o primeiro frame valido do video ou webcam."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret and frame is not None else None


def _frame_to_pil(frame_bgr):
    """Converte frame OpenCV (BGR) para PIL Image (RGB)."""
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


_CHECKS = [
    {
        "id": "face_visible",
        "question": "Is there a human face clearly visible in this image? Answer only yes or no.",
        "fail_keywords": ["no"],
        "warning": "Rosto nao detectado na imagem. Posicione-se de frente para a camera.",
        "weight": 40,
        "blocking": True,
    },
    {
        "id": "lighting",
        "question": "Is the lighting in this image good and bright enough to clearly see the person's skin? Answer only yes or no.",
        "fail_keywords": ["no"],
        "warning": "Iluminacao insuficiente. Use luz natural ou acenda mais luzes.",
        "weight": 25,
        "blocking": False,
    },
    {
        "id": "obstruction",
        "question": "Is the person wearing sunglasses, a face mask, or anything covering their face? Answer only yes or no.",
        "fail_keywords": ["yes"],
        "warning": "Rosto parcialmente coberto (oculos escuros/mascara). Remova para melhor precisao.",
        "weight": 20,
        "blocking": False,
    },
    {
        "id": "distance",
        "question": "Is the person's face close enough to the camera, filling at least a third of the image? Answer only yes or no.",
        "fail_keywords": ["no"],
        "warning": "Rosto muito distante. Aproxime-se (~50cm da camera).",
        "weight": 15,
        "blocking": False,
    },
]


def validate_frame(source, strict=False):
    """
    Valida condicoes de captura usando Moondream2 (local).

    Args:
        source: caminho do video (str) ou indice da camera (int)
        strict: se True, bloqueia execucao se score < 60

    Returns:
        dict com quality_score, warnings, should_proceed, details
    """
    print("  Carregando modelo de visao (Moondream2 local)...")
    model, tokenizer = _load_model()
    if model is None:
        return {"quality_score": -1, "warnings": ["Moondream indisponivel, validacao ignorada."],
                "should_proceed": True, "details": {}}

    frame = _grab_first_frame(source)
    if frame is None:
        return {"quality_score": 0, "warnings": ["Nao foi possivel capturar frame."],
                "should_proceed": False, "details": {}}

    image = _frame_to_pil(frame)
    encoded = model.encode_image(image)

    score = 100
    warnings = []
    details = {}
    has_blocking_fail = False

    print("  Analisando condicoes de captura...")
    for check in _CHECKS:
        answer = model.answer_question(encoded, check["question"], tokenizer).strip().lower()
        failed = any(kw in answer for kw in check["fail_keywords"])
        details[check["id"]] = {"answer": answer, "passed": not failed}

        if failed:
            score -= check["weight"]
            warnings.append(check["warning"])
            if check["blocking"]:
                has_blocking_fail = True

    # Libera memoria apos validacao
    import gc, torch
    del encoded
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    should_proceed = not has_blocking_fail if not strict else (score >= 60)

    return {
        "quality_score": max(0, score),
        "warnings": warnings,
        "should_proceed": should_proceed,
        "details": details,
    }


def print_validation_result(result):
    """Imprime resultado da pre-validacao no terminal."""
    score = result["quality_score"]

    if score == -1:
        print("  Pre-validacao ignorada (modelo indisponivel).")
        return

    if score >= 80:
        status = "OTIMO"
    elif score >= 60:
        status = "ACEITAVEL"
    elif score >= 40:
        status = "RUIM"
    else:
        status = "INADEQUADO"

    print(f"  Qualidade da captura: {score}/100 ({status})")

    for w in result["warnings"]:
        print(f"  ⚠ {w}")

    if not result["should_proceed"]:
        print("  ✖ Condicoes inadequadas para prosseguir. Corrija os problemas acima.")
