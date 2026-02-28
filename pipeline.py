"""
Feature 6: Pipeline Orchestration & Terminal Logging

Ties all modules together in an async execution loop:
  Stream → VAD → STT → Role Manager → Classifier (Transmitter only)

Usage:
    python pipeline.py <path_to.wav>
"""

import asyncio
import sys
from datetime import datetime

from stream_simulator import stream_audio_chunks
from vad_filter import VADFilter
from stt_engine import STTEngine
from speaker_role_manager import SpeakerRoleManager
from onnx_classifier import ONNXClassifier


def log_entry(role: str, speaker_id: str, classification: str, text: str):
    """Print a formatted log line to the terminal."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] | {role}/{speaker_id} | "
        f"Classification: {classification} | "
        f"Text: {text}"
    )


async def run_pipeline(
    wav_path: str,
    simulate_realtime: bool = False,
):
    """
    Main async pipeline loop.

    Flow:
        stream_audio_chunks → VADFilter → STTEngine
        → SpeakerRoleManager → ONNXClassifier (Transmitter only)

    Parameters
    ----------
    wav_path : str
        Path to a 16 kHz mono .wav file.
    simulate_realtime : bool
        Whether to simulate real-time with delays between chunks.
    """
    print("=" * 70)
    print("  Spam Call Classification Pipeline")
    print("=" * 70)
    print(f"  Input : {wav_path}")
    print(f"  Mode  : {'Real-time simulation' if simulate_realtime else 'Fast processing'}")
    print("=" * 70)

    # 1. Initialise all modules
    vad = VADFilter()
    stt = STTEngine()
    role_manager = SpeakerRoleManager()
    classifier = ONNXClassifier()

    # Pre-load models
    print("\n[Pipeline] Loading models ...")
    vad.load()
    stt.load()
    try:
        classifier.load()
        classifier_available = True
    except FileNotFoundError as e:
        print(f"[Pipeline] WARNING: {e}")
        print("[Pipeline] Classifier disabled — will log 'N/A' for classification.")
        classifier_available = False

    print("\n[Pipeline] Starting processing ...\n")
    print("-" * 70)

    # 2. Chain: Stream → VAD → STT
    audio_stream = stream_audio_chunks(
        wav_path, simulate_realtime=simulate_realtime
    )
    vad_stream = vad.filter_chunks(audio_stream)

    # Simple heuristic speaker tracking:
    # Alternate between Speaker 0 and Speaker 1 on each transcription.
    # In a real system, this would come from a diarization model.
    turn_idx = 0
    total_spam = 0
    total_ham = 0
    total_skipped = 0

    async for result in stt.transcribe_stream(vad_stream):
        text = result.text.strip()
        if not text:
            continue

        # Assign speaker (simple alternation heuristic)
        speaker_id = f"Speaker {turn_idx % 2}"
        turn_idx += 1

        # Pass to role manager
        role_info = role_manager.assign_role(speaker_id, text)
        role = role_info["role"]
        is_transmitter = role_info["is_transmitter"]

        # Classify ONLY if speaker is the Transmitter
        if is_transmitter and classifier_available:
            cls_result = classifier.classify(text)
            classification = f"{cls_result['label']} ({cls_result['confidence']:.0%})"
            if cls_result["label"] == "Spam":
                total_spam += 1
            else:
                total_ham += 1
        elif not classifier_available:
            classification = "N/A (no model)"
            total_skipped += 1
        else:
            classification = "Skipped (Receiver)"
            total_skipped += 1

        log_entry(role, speaker_id, classification, text)

    # 3. Summary
    print("-" * 70)
    print(f"\n[Pipeline] Processing complete!")
    print(f"  Total turns    : {turn_idx}")
    print(f"  Spam detected  : {total_spam}")
    print(f"  Ham detected   : {total_ham}")
    print(f"  Skipped        : {total_skipped}")
    print(f"\n  Speaker Summary: {role_manager.get_summary()}")


def main():
    """Entry point with CLI argument parsing."""
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to.wav> [--realtime]")
        print("\nOptions:")
        print("  --realtime   Simulate real-time streaming with delays")
        sys.exit(1)

    wav_path = sys.argv[1]
    simulate_realtime = "--realtime" in sys.argv

    asyncio.run(run_pipeline(wav_path, simulate_realtime=simulate_realtime))


if __name__ == "__main__":
    main()
