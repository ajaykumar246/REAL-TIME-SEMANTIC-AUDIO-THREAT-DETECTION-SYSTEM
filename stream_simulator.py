"""
Feature 1: Stream Simulation & Chunk Buffering

Async generator that reads a local .wav file and yields small
chunks of raw audio bytes to simulate a live WebSocket stream.

Usage (standalone test):
    python stream_simulator.py <path_to_wav>
"""

import asyncio
import wave
import sys

from config import SAMPLE_RATE, CHUNK_DURATION_S, CHANNELS


async def stream_audio_chunks(
    wav_path: str,
    chunk_duration_s: float = CHUNK_DURATION_S,
    sample_rate: int = SAMPLE_RATE,
    simulate_realtime: bool = True,
):
    """
    Async generator that yields raw PCM16 byte chunks from a .wav file.

    Parameters
    ----------
    wav_path : str
        Path to a 16 kHz mono .wav file.
    chunk_duration_s : float
        Duration of each chunk in seconds (default 2.0).
    sample_rate : int
        Expected sample rate (default 16000).
    simulate_realtime : bool
        If True, sleeps between yields to mimic real-time streaming.

    Yields
    ------
    bytes
        Raw PCM16 audio bytes for each chunk.
    """
    with wave.open(wav_path, "rb") as wf:
        # Validate format
        file_sr = wf.getframerate()
        file_ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()

        if file_sr != sample_rate:
            raise ValueError(
                f"Expected sample rate {sample_rate}, got {file_sr}. "
                f"Please resample the audio to {sample_rate} Hz."
            )

        if file_ch != CHANNELS:
            print(
                f"[!] Warning: Expected {CHANNELS} channel(s), got {file_ch}. "
                f"Reading anyway — downstream may need conversion."
            )

        frames_per_chunk = int(sample_rate * chunk_duration_s)
        total_frames = wf.getnframes()
        total_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk

        print(f"[Stream] File: {wav_path}")
        print(f"[Stream] Sample rate: {file_sr} Hz | Channels: {file_ch} | "
              f"Bit depth: {sampwidth * 8}-bit")
        print(f"[Stream] Chunk size: {chunk_duration_s}s ({frames_per_chunk} frames) | "
              f"Total chunks: {total_chunks}")

        chunk_idx = 0
        while True:
            raw_data = wf.readframes(frames_per_chunk)
            if not raw_data:
                break

            chunk_idx += 1
            print(f"[Stream] Yielding chunk {chunk_idx}/{total_chunks} "
                  f"({len(raw_data)} bytes)")

            yield raw_data

            if simulate_realtime:
                await asyncio.sleep(chunk_duration_s)

    print(f"[Stream] Finished — {chunk_idx} chunks yielded.")


# ──────────────────────────── standalone test
async def _test(wav_path: str):
    """Quick test: count and print chunk sizes."""
    chunk_count = 0
    total_bytes = 0
    async for chunk in stream_audio_chunks(wav_path, simulate_realtime=False):
        chunk_count += 1
        total_bytes += len(chunk)
    print(f"\n[Test] {chunk_count} chunks, {total_bytes:,} bytes total.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stream_simulator.py <path_to.wav>")
        sys.exit(1)
    asyncio.run(_test(sys.argv[1]))
