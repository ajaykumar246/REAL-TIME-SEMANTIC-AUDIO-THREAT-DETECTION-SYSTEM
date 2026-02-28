"""
Feature 2: Voice Activity Detection (VAD) Integration

Uses Silero-VAD (PyTorch Hub) to filter out silence from incoming
audio chunks. Only chunks containing human speech are yielded.

Usage (standalone test):
    python vad_filter.py <path_to.wav>
"""

import asyncio
import sys
import struct

import torch
import numpy as np

from config import SAMPLE_RATE, VAD_THRESHOLD


class VADFilter:
    """
    Wraps Silero-VAD for streaming speech detection.

    Loads the model once and provides an async generator that
    filters audio chunks, yielding only those with speech.
    """

    def __init__(self, threshold: float = VAD_THRESHOLD, sample_rate: int = SAMPLE_RATE):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model = None
        self._loaded = False

    def load(self):
        """Load Silero-VAD model from PyTorch Hub."""
        if self._loaded:
            return

        print("[VAD] Loading Silero-VAD model ...")
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()
        self._loaded = True
        print("[VAD] Model loaded (CPU).")

    def _bytes_to_tensor(self, raw_bytes: bytes) -> torch.Tensor:
        """
        Convert raw PCM16 bytes to a float32 torch Tensor in [-1, 1].
        """
        # PCM16 little-endian → int16
        num_samples = len(raw_bytes) // 2
        samples = struct.unpack(f"<{num_samples}h", raw_bytes)
        arr = np.array(samples, dtype=np.float32) / 32768.0
        return torch.from_numpy(arr)

    def get_speech_prob(self, raw_bytes: bytes) -> float:
        """
        Return speech probability for a raw PCM16 chunk.
        """
        if not self._loaded:
            self.load()

        tensor = self._bytes_to_tensor(raw_bytes)

        # Silero-VAD expects 16 kHz mono input
        with torch.no_grad():
            prob = self.model(tensor, self.sample_rate).item()
        return prob

    def reset_state(self):
        """Reset VAD hidden states between files / sessions."""
        if self.model is not None:
            self.model.reset_states()

    async def filter_chunks(self, audio_chunk_gen):
        """
        Async generator: yields only chunks that contain speech.

        Parameters
        ----------
        audio_chunk_gen : async generator
            Yields raw PCM16 bytes (from stream_simulator).

        Yields
        ------
        tuple[bytes, float]
            (raw_audio_bytes, speech_probability)
        """
        if not self._loaded:
            self.load()

        self.reset_state()
        passed = 0
        filtered = 0

        async for chunk in audio_chunk_gen:
            prob = self.get_speech_prob(chunk)

            if prob >= self.threshold:
                passed += 1
                print(f"[VAD] Chunk PASS  (speech prob: {prob:.3f})")
                yield chunk, prob
            else:
                filtered += 1
                print(f"[VAD] Chunk SKIP  (speech prob: {prob:.3f})")

        print(f"[VAD] Done — passed: {passed}, filtered: {filtered}")


# ──────────────────────────── standalone test
async def _test(wav_path: str):
    from stream_simulator import stream_audio_chunks

    vad = VADFilter()
    vad.load()

    speech_chunks = 0
    async for chunk, prob in vad.filter_chunks(
        stream_audio_chunks(wav_path, simulate_realtime=False)
    ):
        speech_chunks += 1

    print(f"\n[Test] {speech_chunks} chunks with speech detected.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vad_filter.py <path_to.wav>")
        sys.exit(1)
    asyncio.run(_test(sys.argv[1]))
