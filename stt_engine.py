"""
Feature 3: Streaming STT (Faster-Whisper)

Transcribes VAD-filtered audio chunks using faster-whisper
with int8 quantization for CPU optimization.

Usage (standalone test):
    python stt_engine.py <path_to.wav>
"""

import asyncio
import sys
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from config import (
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
    avg_logprob: float


class STTEngine:
    """
    Streaming Speech-to-Text engine using faster-whisper.

    Optimized for CPU via int8 compute type and constrained beam search.
    Supports English, Tamil, and auto-detection for Tanglish.
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
        """Load the faster-whisper model."""
        if self._loaded:
            return

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
        """Convert raw PCM16 bytes to float32 numpy array in [-1, 1]."""
        num_samples = len(raw_bytes) // 2
        samples = struct.unpack(f"<{num_samples}h", raw_bytes)
        return np.array(samples, dtype=np.float32) / 32768.0

    def transcribe_chunk(self, raw_bytes: bytes) -> Optional[TranscriptionResult]:
        """
        Transcribe a single audio chunk.

        Parameters
        ----------
        raw_bytes : bytes
            Raw PCM16 audio bytes (16 kHz mono).

        Returns
        -------
        TranscriptionResult or None
            Transcription result, or None if no speech was detected.
        """
        if not self._loaded:
            self.load()

        audio = self._bytes_to_float32(raw_bytes)
        duration_s = len(audio) / SAMPLE_RATE

        segments, info = self.model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=False,       # already filtered by our VAD
            word_timestamps=False,  # not needed, saves CPU
        )

        # Collect all segment texts
        texts = []
        avg_logprobs = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                texts.append(text)
                avg_logprobs.append(seg.avg_log_prob)

        if not texts:
            return None

        full_text = " ".join(texts)
        avg_lp = sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else 0.0

        return TranscriptionResult(
            text=full_text,
            language=info.language,
            language_prob=info.language_probability,
            duration_s=duration_s,
            avg_logprob=avg_lp,
        )

    async def transcribe_stream(self, vad_chunk_gen):
        """
        Async generator: transcribes VAD-filtered chunks.

        Parameters
        ----------
        vad_chunk_gen : async generator
            Yields (raw_bytes, speech_prob) tuples from VADFilter.

        Yields
        ------
        TranscriptionResult
            Transcription for each speech-containing chunk.
        """
        if not self._loaded:
            self.load()

        chunk_idx = 0
        async for raw_bytes, speech_prob in vad_chunk_gen:
            chunk_idx += 1

            # Run transcription (CPU-bound; use run_in_executor for true async)
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


# ──────────────────────────── standalone test
async def _test(wav_path: str):
    from stream_simulator import stream_audio_chunks
    from vad_filter import VADFilter

    vad = VADFilter()
    stt = STTEngine()
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
