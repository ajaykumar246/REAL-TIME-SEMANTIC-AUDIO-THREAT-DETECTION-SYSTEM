"""
Feature 5: ONNX Runtime Classifier

Loads a pre-exported ONNX IndicBERT (MuRIL) model and runs
CPU inference to classify text as Spam or Ham.

Usage (standalone test):
    python onnx_classifier.py
"""

import os
import sys

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from config import (
    ONNX_QUANTIZED_MODEL_PATH,
    ONNX_MODEL_PATH,
    FINE_TUNED_MODEL_DIR,
    TOKENIZER_NAME,
    CLASSIFIER_MAX_LENGTH,
    CLASSIFIER_LABELS,
)


class ONNXClassifier:
    """
    CPU-optimized ONNX Runtime classifier for Spam/Ham detection.

    Uses the quantized ONNX model (int8) for fastest CPU inference.
    Falls back to the unquantized model if quantized version is absent.
    """

    def __init__(
        self,
        model_path: str = None,
        tokenizer_path: str = None,
        max_length: int = CLASSIFIER_MAX_LENGTH,
        labels: list = None,
    ):
        # Determine model path (prefer quantized)
        if model_path is None:
            if os.path.exists(ONNX_QUANTIZED_MODEL_PATH):
                model_path = ONNX_QUANTIZED_MODEL_PATH
            else:
                model_path = ONNX_MODEL_PATH
        self.model_path = model_path

        # Determine tokenizer path (prefer fine-tuned local)
        if tokenizer_path is None:
            if os.path.exists(FINE_TUNED_MODEL_DIR):
                tokenizer_path = FINE_TUNED_MODEL_DIR
            else:
                tokenizer_path = TOKENIZER_NAME
        self.tokenizer_path = tokenizer_path

        self.max_length = max_length
        self.labels = labels or CLASSIFIER_LABELS
        self.session = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """Load the ONNX model and tokenizer."""
        if self._loaded:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"ONNX model not found at '{self.model_path}'. "
                f"Run fine_tune.py then export_onnx.py first."
            )

        print(f"[Classifier] Loading ONNX model: {self.model_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        print(f"[Classifier] Loading tokenizer: {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self._loaded = True
        print("[Classifier] Ready.")

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def classify(self, text: str) -> dict:
        """
        Classify text as Spam or Ham.

        Parameters
        ----------
        text : str
            Input text to classify.

        Returns
        -------
        dict
            {"label": "Spam"/"Ham", "confidence": float, "probabilities": dict}
        """
        if not self._loaded:
            self.load()

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Run inference
        logits = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            },
        )[0]

        probs = self._softmax(logits[0])
        pred_idx = int(np.argmax(probs))
        label = self.labels[pred_idx]
        confidence = float(probs[pred_idx])

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {
                self.labels[i]: float(probs[i]) for i in range(len(self.labels))
            },
        }


# ──────────────────────────── standalone test
def _test():
    """Quick test with sample texts."""
    classifier = ONNXClassifier()

    try:
        classifier.load()
    except FileNotFoundError as e:
        print(f"[!] {e}")
        sys.exit(1)

    test_texts = [
        "Congratulations! You've won a free prize. Call now to claim!",
        "Hey, are you coming to the meeting tomorrow?",
        "Sir, bank loan offer irukku, 2% interest la kedaikum.",
        "Naan late varuvena, 15 minutes la varuven.",
        "URGENT: Your account has been compromised. Verify immediately!",
        "Bro, cricket match paakka pogalama?",
    ]

    print("=" * 70)
    print("  ONNX Classifier — Test")
    print("=" * 70)

    for text in test_texts:
        result = classifier.classify(text)
        print(f"\n  [{result['label']} | {result['confidence']:.2%}]")
        print(f"    Text: \"{text}\"")
        print(f"    Probs: {result['probabilities']}")


if __name__ == "__main__":
    _test()
