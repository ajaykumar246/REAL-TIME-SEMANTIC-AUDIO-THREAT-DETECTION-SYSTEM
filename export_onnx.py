"""
Export the fine-tuned MuRIL model to ONNX format with int8 quantization.

Reads from fine_tuned_model/ and produces model.onnx + model_quantized.onnx.

Usage:
    python export_onnx.py
"""

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

from config import (
    FINE_TUNED_MODEL_DIR,
    ONNX_MODEL_PATH,
    ONNX_QUANTIZED_MODEL_PATH,
    CLASSIFIER_MAX_LENGTH,
    CLASSIFIER_LABELS,
)


def export_to_onnx():
    """Export PyTorch model to ONNX format."""
    print("=" * 60)
    print("  ONNX Export & Quantization")
    print("=" * 60)

    # 1. Load fine-tuned model + tokenizer
    print(f"\n[*] Loading model from {FINE_TUNED_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_DIR)
    model.eval()

    # 2. Create dummy input
    dummy_text = "This is a test sentence for ONNX export."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=CLASSIFIER_MAX_LENGTH,
    )

    # 3. Export to ONNX
    print(f"[*] Exporting to {ONNX_MODEL_PATH} ...")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
    )
    size_mb = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
    print(f"[+] ONNX model saved: {size_mb:.1f} MB")

    # 4. Quantize (int8 dynamic quantization for CPU)
    print(f"[*] Quantizing to {ONNX_QUANTIZED_MODEL_PATH} ...")
    quantize_dynamic(
        model_input=ONNX_MODEL_PATH,
        model_output=ONNX_QUANTIZED_MODEL_PATH,
        weight_type=QuantType.QInt8,
    )
    q_size_mb = os.path.getsize(ONNX_QUANTIZED_MODEL_PATH) / (1024 * 1024)
    print(f"[+] Quantized model saved: {q_size_mb:.1f} MB "
          f"({(1 - q_size_mb / size_mb) * 100:.0f}% smaller)")

    # 5. Validate the exported model
    print("\n[*] Validating ONNX model ...")
    validate_onnx(tokenizer)

    print("\n[✓] ONNX export complete!")


def validate_onnx(tokenizer):
    """Run a test inference with the quantized ONNX model."""
    session = ort.InferenceSession(
        ONNX_QUANTIZED_MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )

    test_texts = [
        "Congratulations! You've won a free iPhone. Call now!",
        "Hey, are you coming to the meeting tomorrow?",
        "Sir ungalukku special loan offer irukku, bank-la apply pannunga.",
        "Naan 5 minutes la varuven, wait pannu.",
    ]

    for text in test_texts:
        inputs = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=CLASSIFIER_MAX_LENGTH,
        )

        logits = session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            },
        )[0]

        probs = softmax(logits[0])
        pred_idx = int(np.argmax(probs))
        label = CLASSIFIER_LABELS[pred_idx]
        confidence = float(probs[pred_idx])
        print(f"    [{label} {confidence:.2%}] {text[:60]}...")


def softmax(x):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


if __name__ == "__main__":
    export_to_onnx()
