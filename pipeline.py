"""
Feature 6: Pipeline Orchestration & Logging

Ties all modules together in an async execution loop:
  Stream → VAD → STT → Classifier → Log

Usage:
    python pipeline.py <path_to.wav>
"""

import asyncio
import os
import sys
from datetime import datetime

from stream_simulator import stream_audio_chunks
from vad_filter import VADFilter
from stt_engine import create_stt_engine
from classifier import ONNXClassifier as MLClassifier

# Log file path
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_output.log")


def log_entry(segment_num: int, classification: str, confidence: float,
              text: str, log_lines: list):
    """Print and store a formatted log line."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"[{timestamp}] | Segment {segment_num} | "
        f"{classification} ({confidence:.0%}) | "
        f"Text: {text}"
    )
    print(line)
    log_lines.append(line)


async def run_pipeline(
    wav_path: str,
    simulate_realtime: bool = False,
):
    """
    Main async pipeline loop.

    Flow:
        stream_audio_chunks → VADFilter (accumulate) → STT → Classifier

    Every transcribed segment is classified directly — no speaker
    role filtering, since accumulated segments contain mixed speakers.

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
    stt = create_stt_engine()
    classifier = MLClassifier()

    # Pre-load models
    print("\n[Pipeline] Loading models ...")
    vad.load()
    stt.load()
    try:
        classifier.load()
        classifier_available = True
    except FileNotFoundError as e:
        print(f"[Pipeline] WARNING: {e}")
        print("[Pipeline] Classifier disabled — will log 'N/A'.")
        classifier_available = False

    print("\n[Pipeline] Starting processing ...\n")
    print("-" * 70)

    # 2. Chain: Stream → VAD (accumulate) → STT → Classify
    audio_stream = stream_audio_chunks(
        wav_path, simulate_realtime=simulate_realtime
    )
    vad_stream = vad.filter_chunks(audio_stream)

    segment_num = 0
    total_spam = 0
    total_ham = 0
    all_texts = []
    log_lines = []

    async for result in stt.transcribe_stream(vad_stream):
        text = result.text.strip()
        if not text:
            continue

        segment_num += 1
        all_texts.append(text)

        # Classify every segment
        if classifier_available:
            cls_result = classifier.classify(text)
            label = cls_result["label"]
            confidence = cls_result["confidence"]

            if label == "Spam":
                total_spam += 1
            else:
                total_ham += 1

            log_entry(segment_num, label, confidence, text, log_lines)
        else:
            log_entry(segment_num, "N/A", 0.0, text, log_lines)

    # 3. Overall call verdict
    print("-" * 70)

    # If ANY segment is spam, the call is spam
    if total_spam > 0:
        call_verdict = "🚨 SPAM CALL DETECTED"
    elif total_ham > 0:
        call_verdict = "✅ Legitimate Call (Ham)"
    else:
        call_verdict = "⚠️ No speech classified"

    print(f"\n{'=' * 70}")
    print(f"  RESULT: {call_verdict}")
    print(f"{'=' * 70}")
    print(f"  Total segments : {segment_num}")
    print(f"  Spam segments  : {total_spam}")
    print(f"  Ham segments   : {total_ham}")
    print(f"{'=' * 70}")

    # 4. Write full log to file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Pipeline Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input: {wav_path}\n")
        f.write(f"Result: {call_verdict}\n")
        f.write(f"Segments: {segment_num} (Spam: {total_spam}, Ham: {total_ham})\n")
        f.write("=" * 70 + "\n\n")

        for line in log_lines:
            f.write(line + "\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("FULL TRANSCRIPTION:\n")
        f.write("=" * 70 + "\n")
        for i, text in enumerate(all_texts, 1):
            f.write(f"\n[Segment {i}]\n{text}\n")

    print(f"\n  Full log saved: {LOG_FILE}")


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
