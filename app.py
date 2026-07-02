"""
Real-Time Meeting Assistant - Flask + SocketIO backend with live frontend.
"""
import threading
from collections import deque

import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from src.audio_input.recorder import AudioRecorder
from src.speech_to_text.speech_to_text import SpeechToText
from src.nlp_processing.topic_extraction import extract_topics_tfidf, extract_topics_lda
from src.nlp_processing.action_items import extract_action_items
from src.summarizer.summary import summarize_meeting

app = Flask(__name__)
app.config["SECRET_KEY"] = "meeting-assistant-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Global state
recorder = None
asr = None
audio_buffer = deque(maxlen=10000)  # ~10s at 16kHz
buffer_lock = threading.Lock()
SAMPLE_RATE = 16000


def process_audio_callback(audio_chunk: np.ndarray) -> None:
    """Callback: accumulate audio chunks into buffer."""
    with buffer_lock:
        audio_buffer.extend(audio_chunk.tolist())


def _transcribe_buffer() -> str:
    """Transcribe accumulated audio and clear buffer."""
    with buffer_lock:
        if not audio_buffer:
            return ""
        audio = np.array(list(audio_buffer), dtype=np.float32)
        audio_buffer.clear()
    if len(audio) < SAMPLE_RATE * 0.5:  # Require at least 0.5s
        return ""
    try:
        return asr.transcribe(audio, SAMPLE_RATE)
    except Exception as e:
        print(f"ASR error: {e}")
        return ""


def _run_nlp(text: str) -> dict:
    """Extract topics, action items, and summary from text."""
    if not text.strip():
        return {"transcript": "", "topics": [], "action_items": [], "summary": ""}
    topics_tfidf = extract_topics_tfidf([text], top_k=5)
    topics_lda = extract_topics_lda([text], n_topics=3, n_words=4)
    actions = extract_action_items(text)
    topics = [t[0] for t in topics_tfidf]
    if not topics and topics_lda:
        topics = [t[0] for t_list in topics_lda for t in t_list[:2]]
    summary = summarize_meeting([text], topics[:5], actions)
    return {
        "transcript": text,
        "topics": topics[:5],
        "action_items": actions,
        "summary": summary,
    }


@app.route("/")
def index():
    """Serve the main frontend."""
    return render_template("index.html")


@socketio.on("connect")
def on_connect():
    """Handle client connection."""
    emit("connected", {"status": "ok"})


@socketio.on("start_recording")
def on_start():
    """Start microphone recording and ASR."""
    global recorder, asr
    if asr is None:
        try:
            asr = SpeechToText()
        except Exception as e:
            emit("error", {"message": f"ASR init failed: {e}"})
            return
    if recorder is None:
        recorder = AudioRecorder(sample_rate=SAMPLE_RATE)
    with buffer_lock:
        audio_buffer.clear()
    recorder.start(process_audio_callback)
    emit("recording_started", {})


@socketio.on("stop_recording")
def on_stop():
    """Stop recording, transcribe buffer, and emit NLP results."""
    global recorder
    if recorder:
        recorder.stop()
    text = _transcribe_buffer()
    result = _run_nlp(text)
    emit("results", result)
    emit("recording_stopped", {})


@socketio.on("process_transcript")
def on_process(data):
    """Process manually provided transcript."""
    text = data.get("text", "")
    result = _run_nlp(text)
    emit("results", result)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
