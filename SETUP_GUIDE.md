# Setup Guide: Real-Time Semantic Audio Threat Detection System

This guide will walk you through setting up and running the Threat Detection Web UI on any new machine. Since the model has already been fine-tuned, you **do not** need to retrain it.

## Prerequisites

Before starting, ensure your system has the following installed:

1. **Python 3.9 or higher**
2. **FFmpeg** (Required for the web server to convert uploaded audio files to `16kHz WAV` format automatically).
   - **Windows:** Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), extract it, and add the `bin` folder to your system's `PATH` environment variable.
   - **macOS:** `brew install ffmpeg`
   - **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install ffmpeg`

## Step 1: Clone or Copy the Repository
Bring this entire project folder to the new machine. Ensure that the `fine_tuned_model/` directory remains intact inside the project root, as this contains the pre-trained MuRIL classifier.

## Step 2: Install Python Dependencies
Open your terminal or command prompt, navigate to the project root directory, and run the following command to install all necessary libraries:

```bash
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` file has been optimized for deployment and does not include heavy training dependencies.

## Step 3: Configure the Groq API Key
The pipeline uses Groq's blazing-fast Whisper API for Speech-to-Text conversion.

1. Go to [Groq Console](https://console.groq.com/) and create a free account.
2. Generate a free API Key.
3. Open `config.py` in your project folder.
4. Paste your API key into the `GROQ_API_KEY` string:
   ```python
   GROQ_API_KEY = "gsk_YourAPIKeyHere..."
   ```

## Step 4: Run the Web Server
Launch the FastAPI web server, which handles the pipeline execution and serves the frontend Visualizer.

```bash
python app.py
```

You should see output indicating that the models are loading and the server has started.

## Step 5: Test the Pipeline
1. Open your web browser.
2. Navigate to: `http://localhost:8000`
3. Drag and drop any audio file (e.g., `.wav`, `.mp3`) into the upload zone.
4. Click **Start Pipeline**.
5. Watch the live visualization as the audio is chunked, transcribed, and classified in real-time!

---

### Folder Structure Overview
For your reference, here is the clean deployment structure:
- `app.py`: The FastAPI server and WebSocket handler.
- `pipeline.py`: The core orchestrator chaining VAD -> STT -> Classifier.
- `config.py`: System configurations and Spam Keyword lists.
- `static/`: Frontend HTML/CSS/JS files for the Web UI.
- `fine_tuned_model/`: The pre-trained PyTorch Classifier.
- `training_scripts/`: (Optional) Contains dataset preparation and fine-tuning scripts; this is explicitly ignored for deployment.
