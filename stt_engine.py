"""
Feature 3: Streaming STT — Dual Mode (Groq API / Local Faster-Whisper)

Supports two backends:
  - "api"   → Groq Whisper API (whisper-large-v3-turbo, fast + accurate)
  - "local" → Faster-Whisper local (int8, CPU, offline)

Usage (standalone test):
    python stt_engine.py <path_to.wav>
"""

import asyncio
import io
import sys
import struct
import wave
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    STT_MODE,
    GROQ_API_KEY,
    GROQ_MODEL,
    WHISPER_MODEL_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_BEAM_SIZE,
    WHISPER_LANGUAGE,
    SAMPLE_RATE,
)


@dataclass
class TranscriptionResult:
    """Container for a single transcription output."""
    text: str
    language: str
    language_prob: float
    duration_s: float


# ════════════════════════════════════════════════
#  Groq API Backend
# ════════════════════════════════════════════════

class GroqSTTEngine:
    """
    Speech-to-Text via Groq's Whisper API.
    Uses whisper-large-v3-turbo for best accuracy on Tamil/Tanglish/English.
    ~10× real-time speed, free tier available.
    """

    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "and set it in config.py or as an environment variable."
            )

        from groq import Groq
        self.client = Groq(api_key=self.api_key)
        self._loaded = True
        print(f"[STT] Groq API ready (model: {self.model})")

    @staticmethod
    def _bytes_to_wav_buffer(raw_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> io.BytesIO:
        """Wrap raw PCM16 bytes into an in-memory WAV file for the API."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(raw_bytes)
        buf.seek(0)
        buf.name = "chunk.wav"  # Groq API needs a filename
        return buf

    def transcribe_chunk(self, raw_bytes: bytes) -> Optional[TranscriptionResult]:
        if not self._loaded:
            self.load()

        duration_s = (len(raw_bytes) / 2) / SAMPLE_RATE
        wav_buf = self._bytes_to_wav_buffer(raw_bytes)

        try:
            response = self.client.audio.transcriptions.create(
                file=wav_buf,
                model=self.model,
                language=WHISPER_LANGUAGE if WHISPER_LANGUAGE else None,
                response_format="verbose_json",
                prompt=(
                    "This is a phone call conversation in Tamil, English, "
                    "or Tanglish (Tamil-English code-mixed). "
                    "Common words: vanakkam, sir, madam, hello, offer, bank, "
                    "loan, nandri, sollunga, pannunga, irukku, kedaikum."
                ),
            )

            text = response.text.strip() if response.text else ""
            language = getattr(response, "language", "unknown") or "unknown"

            if not text:
                return None

            return TranscriptionResult(
                text=text,
                language=language,
                language_prob=1.0,   # API doesn't return prob
                duration_s=duration_s,
            )
        except Exception as e:
            print(f"[STT] Groq API error: {e}")
            return None

    async def transcribe_stream(self, vad_chunk_gen):
        if not self._loaded:
            self.load()

        chunk_idx = 0
        async for raw_bytes, speech_prob in vad_chunk_gen:
            chunk_idx += 1

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.transcribe_chunk, raw_bytes
            )

            if result and result.text.strip():
                print(
                    f"[STT] Chunk {chunk_idx} | Lang: {result.language} | "
                    f'Text: "{result.text}"'
                )
                yield result
            else:
                print(f"[STT] Chunk {chunk_idx} | (no speech transcribed)")


# ════════════════════════════════════════════════
#  Local Faster-Whisper Backend
# ════════════════════════════════════════════════

class LocalSTTEngine:
    """
    Local Speech-to-Text using faster-whisper (CTranslate2).
    CPU-optimized with int8 quantization. Works fully offline.
    """

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        device: str = WHISPER_DEVICE,
        beam_size: int = WHISPER_BEAM_SIZE,
        language: Optional[str] = WHISPER_LANGUAGE,
    ):
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.beam_size = beam_size
        self.language = language
        self.model = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        from faster_whisper import WhisperModel

        print(f"[STT] Loading faster-whisper model: {self.model_size} "
              f"(compute_type={self.compute_type}, device={self.device})")
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=4,
        )
        self._loaded = True
        print("[STT] Model loaded.")

    @staticmethod
    def _bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
        num_samples = len(raw_bytes) // 2
        samples = struct.unpack(f"<{num_samples}h", raw_bytes)
        return np.array(samples, dtype=np.float32) / 32768.0

    def transcribe_chunk(self, raw_bytes: bytes) -> Optional[TranscriptionResult]:
        if not self._loaded:
            self.load()

        audio = self._bytes_to_float32(raw_bytes)
        duration_s = len(audio) / SAMPLE_RATE

        segments, info = self.model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=False,
            word_timestamps=False,
        )

        texts = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                texts.append(text)

        if not texts:
            return None

        return TranscriptionResult(
            text=" ".join(texts),
            language=info.language,
            language_prob=info.language_probability,
            duration_s=duration_s,
        )

    async def transcribe_stream(self, vad_chunk_gen):
        if not self._loaded:
            self.load()

        chunk_idx = 0
        async for raw_bytes, speech_prob in vad_chunk_gen:
            chunk_idx += 1

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.transcribe_chunk, raw_bytes
            )

            if result and result.text.strip():
                print(
                    f"[STT] Chunk {chunk_idx} | Lang: {result.language} "
                    f"({result.language_prob:.0%}) | "
                    f'Text: "{result.text}"'
                )
                yield result
            else:
                print(f"[STT] Chunk {chunk_idx} | (no speech transcribed)")


# ════════════════════════════════════════════════
#  Factory: pick the right engine
# ════════════════════════════════════════════════

def create_stt_engine(mode: str = STT_MODE):
    """
    Create the appropriate STT engine based on config.

    Parameters
    ----------
    mode : str
        "api" for Groq Whisper API, "local" for Faster-Whisper.
    """
    if mode == "api":
        print("[STT] Using Groq Whisper API (fast + accurate)")
        return GroqSTTEngine()
    elif mode == "local":
        print("[STT] Using local Faster-Whisper (offline, CPU)")
        return LocalSTTEngine()
    else:
        raise ValueError(f"Unknown STT_MODE: {mode}. Use 'api' or 'local'.")


# For backward compatibility
STTEngine = create_stt_engine


# ──────────────────────────── standalone test
async def _test(wav_path: str):
    from stream_simulator import stream_audio_chunks
    from vad_filter import VADFilter

    vad = VADFilter()
    stt = create_stt_engine()
    stt.load()

    count = 0
    async for result in stt.transcribe_stream(
        vad.filter_chunks(stream_audio_chunks(wav_path, simulate_realtime=False))
    ):
        count += 1
        print(f"  [{count}] ({result.language}) {result.text}")

    print(f"\n[Test] {count} transcriptions produced.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stt_engine.py <path_to.wav>")
        sys.exit(1)
    asyncio.run(_test(sys.argv[1]))
