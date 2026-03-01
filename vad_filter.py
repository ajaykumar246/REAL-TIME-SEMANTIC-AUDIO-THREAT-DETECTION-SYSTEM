"""
Feature 2: Voice Activity Detection (VAD) with Speech Accumulation

Uses Silero-VAD to detect speech, then accumulates consecutive speech
chunks into a single buffer. Only sends to STT when a silence gap is
detected (natural sentence boundary). This gives Whisper full sentences
instead of 2-second fragments.

Usage (standalone test):
    python vad_filter.py <path_to.wav>
"""

import asyncio
import sys
import struct

import torch
import numpy as np

from config import SAMPLE_RATE, VAD_THRESHOLD


# Silero-VAD v5 requires exactly this many samples per call
VAD_WINDOW_SAMPLES = 512  # for 16 kHz


class VADFilter:
    """
    Wraps Silero-VAD for streaming speech detection with accumulation.

    Instead of yielding every speech chunk individually, it buffers
    consecutive speech chunks and yields them as one combined segment
    when silence is detected. This produces longer, more meaningful
    audio segments for better STT accuracy.
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

    def _bytes_to_float_array(self, raw_bytes: bytes) -> np.ndarray:
        """Convert raw PCM16 bytes to float32 numpy array in [-1, 1]."""
        num_samples = len(raw_bytes) // 2
        samples = struct.unpack(f"<{num_samples}h", raw_bytes)
        return np.array(samples, dtype=np.float32) / 32768.0

    def get_speech_prob(self, raw_bytes: bytes) -> float:
        """
        Return speech probability for a raw PCM16 chunk.

        Splits into 512-sample windows, returns max probability.
        """
        if not self._loaded:
            self.load()

        audio = self._bytes_to_float_array(raw_bytes)
        total_samples = len(audio)

        if total_samples == 0:
            return 0.0

        probabilities = []

        for start in range(0, total_samples, VAD_WINDOW_SAMPLES):
            end = start + VAD_WINDOW_SAMPLES
            window = audio[start:end]

            if len(window) < VAD_WINDOW_SAMPLES:
                window = np.pad(window, (0, VAD_WINDOW_SAMPLES - len(window)))

            tensor = torch.from_numpy(window)

            with torch.no_grad():
                prob = self.model(tensor, self.sample_rate).item()
                probabilities.append(prob)

        return max(probabilities) if probabilities else 0.0

    def reset_state(self):
        """Reset VAD hidden states between files / sessions."""
        if self.model is not None:
            self.model.reset_states()

    async def filter_chunks(self, audio_chunk_gen):
        """
        Async generator: accumulates consecutive speech chunks and
        yields combined audio segments on silence boundaries.

        Instead of yielding every individual chunk, it:
        1. Buffers consecutive speech chunks (VAD prob >= threshold)
        2. When silence is detected (VAD prob < threshold), flushes
           the buffer as ONE combined audio segment
        3. At stream end, flushes any remaining buffer

        This gives STT complete sentences instead of fragments.

        Parameters
        ----------
        audio_chunk_gen : async generator
            Yields raw PCM16 bytes (from stream_simulator).

        Yields
        ------
        tuple[bytes, float]
            (combined_audio_bytes, avg_speech_probability)
        """
        if not self._loaded:
            self.load()

        self.reset_state()

        speech_buffer = bytearray()   # accumulate speech bytes
        speech_probs = []             # track probabilities
        passed = 0
        filtered = 0
        segments_yielded = 0

        # Max buffer = 10 seconds (flush for more granular segments)
        MAX_BUFFER_S = 10.0
        max_buffer_bytes = int(MAX_BUFFER_S * self.sample_rate * 2)

        async for chunk in audio_chunk_gen:
            prob = self.get_speech_prob(chunk)

            if prob >= self.threshold:
                # Speech detected — add to buffer
                speech_buffer.extend(chunk)
                speech_probs.append(prob)
                buffer_s = len(speech_buffer) / 2 / self.sample_rate
                print(f"[VAD] Chunk BUFFER (speech prob: {prob:.3f}, "
                      f"buffer: {buffer_s:.1f}s)")
                passed += 1

                # Flush if buffer exceeds max duration
                if len(speech_buffer) >= max_buffer_bytes:
                    avg_prob = sum(speech_probs) / len(speech_probs)
                    duration = len(speech_buffer) / 2 / self.sample_rate
                    segments_yielded += 1
                    print(f"[VAD] FLUSH segment {segments_yielded} "
                          f"({duration:.1f}s, avg prob: {avg_prob:.3f}) [max reached]")
                    yield bytes(speech_buffer), avg_prob
                    speech_buffer.clear()
                    speech_probs.clear()
            else:
                # Silence — flush accumulated buffer if any
                if speech_buffer:
                    avg_prob = sum(speech_probs) / len(speech_probs)
                    duration = len(speech_buffer) / 2 / self.sample_rate
                    segments_yielded += 1
                    print(f"[VAD] FLUSH segment {segments_yielded} "
                          f"({duration:.1f}s, avg prob: {avg_prob:.3f})")
                    yield bytes(speech_buffer), avg_prob
                    speech_buffer.clear()
                    speech_probs.clear()

                filtered += 1
                print(f"[VAD] Chunk SKIP  (speech prob: {prob:.3f})")

        # Flush remaining buffer at end of stream
        if speech_buffer:
            avg_prob = sum(speech_probs) / len(speech_probs)
            duration = len(speech_buffer) / 2 / self.sample_rate
            segments_yielded += 1
            print(f"[VAD] FLUSH final segment {segments_yielded} "
                  f"({duration:.1f}s, avg prob: {avg_prob:.3f})")
            yield bytes(speech_buffer), avg_prob

        print(f"[VAD] Done — chunks: {passed} speech + {filtered} silence "
              f"→ {segments_yielded} segments yielded")


# ──────────────────────────── standalone test
async def _test(wav_path: str):
    from stream_simulator import stream_audio_chunks

    vad = VADFilter()
    vad.load()

    segments = 0
    async for audio, prob in vad.filter_chunks(
        stream_audio_chunks(wav_path, simulate_realtime=False)
    ):
        segments += 1
        duration = len(audio) / 2 / SAMPLE_RATE
        print(f"  Segment {segments}: {duration:.1f}s (prob: {prob:.3f})")

    print(f"\n[Test] {segments} speech segments detected.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vad_filter.py <path_to.wav>")
        sys.exit(1)
    asyncio.run(_test(sys.argv[1]))
