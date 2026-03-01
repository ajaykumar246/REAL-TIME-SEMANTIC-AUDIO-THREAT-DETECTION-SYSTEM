"""
FastAPI Web Server for Spam Call Classification Pipeline.

Provides:
  - GET  /            → Serves the web UI
  - POST /upload      → Upload audio file
  - WS   /ws/process  → WebSocket for real-time pipeline results

Usage:
    python app.py
"""

import asyncio
import json
import os
import shutil
import tempfile
import traceback
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from stream_simulator import stream_audio_chunks
from vad_filter import VADFilter
from stt_engine import create_stt_engine
from classifier import ONNXClassifier as MLClassifier

app = FastAPI(title="Spam Call Classifier")

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pre-load models at startup
vad = VADFilter()
stt = create_stt_engine()
classifier = MLClassifier()


@app.on_event("startup")
async def startup():
    """Load models at server start."""
    vad.load()
    stt.load()
    try:
        classifier.load()
    except FileNotFoundError:
        print("[WARNING] Classifier model not found — classification disabled.")


@app.get("/")
async def index():
    """Serve the main UI page."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an audio file and save it for processing.
    Returns a file ID for the WebSocket to reference.
    """
    # Save to uploads dir
    file_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, file_id)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size = os.path.getsize(file_path) / 1024  # KB

    return {
        "file_id": file_id,
        "filename": file.filename,
        "size_kb": round(file_size, 1),
    }


@app.websocket("/ws/process")
async def websocket_process(ws: WebSocket):
    """
    WebSocket endpoint for real-time pipeline processing.

    Client sends: {"file_id": "..."}
    Server streams events:
        - {"type": "status", "message": "..."}
        - {"type": "chunk", "num": N, "total": M, "duration": "5.0s"}
        - {"type": "vad", "action": "buffer"|"flush"|"skip", "buffer_s": X}
        - {"type": "stt", "segment": N, "text": "...", "lang": "..."}
        - {"type": "classify", "segment": N, "label": "...", "confidence": X, ...}
        - {"type": "result", "verdict": "...", "segments": [...]}
    """
    await ws.accept()

    try:
        # Wait for client to send file_id
        data = await ws.receive_text()
        msg = json.loads(data)
        file_id = msg.get("file_id", "")

        file_path = os.path.join(UPLOAD_DIR, file_id)
        if not os.path.exists(file_path):
            await ws.send_json({"type": "error", "message": "File not found"})
            await ws.close()
            return

        # Check if it's a WAV file; if not, try to convert with ffmpeg
        if not file_id.lower().endswith(".wav"):
            await ws.send_json({"type": "status", "message": "Converting audio to WAV (16kHz mono)..."})
            wav_path = file_path + ".wav"
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", file_path,
                "-ar", "16000", "-ac", "1", wav_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            if proc.returncode != 0:
                await ws.send_json({"type": "error", "message": "Failed to convert audio. Please upload a WAV file."})
                await ws.close()
                return
            file_path = wav_path

        await ws.send_json({"type": "status", "message": "Models loaded. Starting pipeline..."})
        await asyncio.sleep(0.1)

        # Run pipeline with streaming events
        segments = []
        segment_num = 0
        total_spam = 0
        total_ham = 0

        # Stream → VAD → STT
        audio_stream = stream_audio_chunks(file_path, simulate_realtime=False)

        # Override VAD to send events
        vad.reset_state()
        chunk_idx = 0

        async for audio_bytes, avg_prob in vad.filter_chunks(audio_stream):
            segment_num += 1
            duration_s = len(audio_bytes) / 2 / 16000

            await ws.send_json({
                "type": "vad",
                "action": "flush",
                "segment": segment_num,
                "duration_s": round(duration_s, 1),
                "speech_prob": round(avg_prob, 3),
            })
            await asyncio.sleep(0.05)

            # STT
            await ws.send_json({
                "type": "status",
                "message": f"Transcribing segment {segment_num} ({duration_s:.1f}s)...",
            })

            result = stt.transcribe_chunk(audio_bytes)

            if result and result.text.strip():
                text = result.text.strip()
                lang = result.language

                await ws.send_json({
                    "type": "stt",
                    "segment": segment_num,
                    "text": text,
                    "language": lang,
                })
                await asyncio.sleep(0.05)

                # Classify
                await ws.send_json({
                    "type": "status",
                    "message": f"Classifying segment {segment_num}...",
                })

                cls_result = classifier.classify(text)
                label = cls_result["label"]
                confidence = cls_result["confidence"]
                method = cls_result.get("method", "model")
                keywords = cls_result.get("keywords_found", [])

                if label == "Spam":
                    total_spam += 1
                else:
                    total_ham += 1

                seg_data = {
                    "segment": segment_num,
                    "text": text,
                    "language": lang,
                    "label": label,
                    "confidence": round(confidence, 3),
                    "method": method,
                    "keywords": keywords,
                    "duration_s": round(duration_s, 1),
                }

                if "spam_sentence" in cls_result:
                    seg_data["spam_sentence"] = cls_result["spam_sentence"]

                segments.append(seg_data)

                await ws.send_json({"type": "classify", **seg_data})
                await asyncio.sleep(0.1)

        # Final verdict
        if total_spam > 0:
            verdict = "SPAM"
        elif total_ham > 0:
            verdict = "HAM"
        else:
            verdict = "UNKNOWN"

        await ws.send_json({
            "type": "result",
            "verdict": verdict,
            "total_segments": segment_num,
            "spam_segments": total_spam,
            "ham_segments": total_ham,
            "segments": segments,
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
