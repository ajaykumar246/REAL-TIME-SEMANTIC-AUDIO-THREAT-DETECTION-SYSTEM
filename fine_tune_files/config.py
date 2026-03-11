"""
Central configuration for the Spam Call Classification Pipeline.
All tuneable constants live here for easy modification.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FINE_TUNED_MODEL_DIR = os.path.join(BASE_DIR, "fine_tuned_model")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "model.onnx")
ONNX_QUANTIZED_MODEL_PATH = os.path.join(BASE_DIR, "model_quantized.onnx")

# ──────────────────────────────────────────────
# Audio / Streaming
# ──────────────────────────────────────────────
SAMPLE_RATE = 16_000          # 16 kHz mono
CHUNK_DURATION_S = 2.0        # seconds per chunk
CHANNELS = 1                  # mono audio

# ──────────────────────────────────────────────
# Voice Activity Detection (Silero-VAD)
# ──────────────────────────────────────────────
VAD_THRESHOLD = 0.5           # speech probability threshold

# ──────────────────────────────────────────────
# Speech-to-Text (Faster-Whisper)
# ──────────────────────────────────────────────
WHISPER_MODEL_SIZE = "small"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_DEVICE = "cpu"
WHISPER_BEAM_SIZE = 3
WHISPER_LANGUAGE = None        # None = auto-detect (en / ta)

# ──────────────────────────────────────────────
# Classifier (ONNX / MuRIL)
# ──────────────────────────────────────────────
TOKENIZER_NAME = "google/muril-base-cased"
CLASSIFIER_MAX_LENGTH = 128
CLASSIFIER_LABELS = ["Ham", "Spam"]

# ──────────────────────────────────────────────
# Speaker Role Manager
# ──────────────────────────────────────────────
SPAM_KEYWORDS = [
    # English
    "offer", "bank", "sir", "madam", "congratulations", "win", "calling",
    "prize", "credit", "loan", "insurance", "scheme", "limited time",
    "urgent", "verify", "account", "free", "selected", "lucky",
    # Tamil / Tanglish
    "vangalam", "panam", "thittam", "bank", "illa", "chance",
    "ungalukku", "jeyicheenga", "kadanai", "thogai",
]

# ──────────────────────────────────────────────
# Fine-tuning
# ──────────────────────────────────────────────
FINETUNE_EPOCHS = 3
FINETUNE_BATCH_SIZE = 16
FINETUNE_LEARNING_RATE = 2e-5
FINETUNE_MAX_LENGTH = 128
FINETUNE_TEST_SIZE = 0.2
