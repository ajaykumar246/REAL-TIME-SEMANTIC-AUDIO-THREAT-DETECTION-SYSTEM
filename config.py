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


# ──────────────────────────────────────────────
# Audio / Streaming
# ──────────────────────────────────────────────
SAMPLE_RATE = 16_000          # 16 kHz mono
CHUNK_DURATION_S = 5.0        # seconds per chunk (larger = better STT accuracy)
CHANNELS = 1                  # mono audio

# ──────────────────────────────────────────────
# Voice Activity Detection (Silero-VAD)
# ──────────────────────────────────────────────
VAD_THRESHOLD = 0.5           # speech probability threshold

# ──────────────────────────────────────────────
# Speech-to-Text
# ──────────────────────────────────────────────
# Mode: "api" (Groq, fast + accurate) or "local" (Faster-Whisper, offline)
STT_MODE = "api"

# --- Groq API (used when STT_MODE = "api") ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # set via env or paste here
GROQ_MODEL = "whisper-large-v3-turbo"               # best speed/accuracy

# --- Faster-Whisper local (used when STT_MODE = "local") ---
WHISPER_MODEL_SIZE = "base"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_DEVICE = "cpu"
WHISPER_BEAM_SIZE = 3
WHISPER_LANGUAGE = None        # Auto-detect language (avoids Whisper hallucinations)
LLM_MODEL = "llama-3.3-70b-versatile"  # High accuracy translation/transliteration

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
    "ungalukku", "jeyicheenga", "kadanai", "thogai", "sir", "madam",
    # Native Tamil Script (Since we are bypassing translation)
    "ஆபர்", "பேங்க்", "சார்", "மேடம்", "வாழ்த்துக்கள்", "வெற்றி", "அழைப்பு",
    "பரிசு", "கிரெடிட்", "லோன்", "காப்பீடு", "திட்டம்", "குறுகிய நேரம்",
    "உடனடி", "சரிபார்க்க", "கணக்கு", "இலவசம்", "தேர்ந்தெடுக்கப்",
    "பணம்", "வாங்கலாம்", "ஜெயிச்சுருக்கீங்க", "உங்களுக்கு"
]

# ──────────────────────────────────────────────
# Fine-tuning
# ──────────────────────────────────────────────
FINETUNE_EPOCHS = 3
FINETUNE_BATCH_SIZE = 16
FINETUNE_LEARNING_RATE = 2e-5
FINETUNE_MAX_LENGTH = 128
FINETUNE_TEST_SIZE = 0.2
