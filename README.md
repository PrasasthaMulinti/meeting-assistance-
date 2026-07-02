# Real-Time AI Meeting Assistant

A Python-based meeting assistant that captures live audio, transcribes speech, extracts topics using TF-IDF + LDA, and detects action items using rule-based keyword detection and POS tagging.

## Features

- **Live microphone capture** – Record audio in real time
- **Speech-to-text** – Pretrained Wav2Vec2 ASR model
- **Audio preprocessing** – MFCC extraction support
- **NLP preprocessing** – Tokenization, stopword removal, lemmatization
- **Topic extraction** – TF-IDF and LDA
- **Action item detection** – Rule-based keywords + POS tagging
- **Live frontend** – Transcript, topics, action items, and summary display

## Project Structure

```
Meeting_Assistant/
├── app.py                 # Flask + SocketIO main app
├── requirements.txt
├── templates/
│   └── index.html         # Frontend UI
└── src/
    ├── audio_input/       # Microphone capture
    ├── speech_to_text/    # ASR (Wav2Vec2)
    ├── nlp_processing/    # Topic extraction, action items
    ├── summarizer/        # Meeting summary
    └── utils/             # Helpers, MFCC preprocessing
```

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv meeting_env
   meeting_env\Scripts\activate   # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data (run once):

   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
   ```

4. On Windows, if PyAudio fails, try:

   ```bash
   pip install pipwin
   pipwin install pyaudio
   ```

## Run

```bash
python app.py
```

Open http://localhost:5000 in your browser.

## Usage

- **Start Recording** – Begin capturing microphone audio
- **Stop Recording** – Stop capture, transcribe, and run NLP
- **Process Transcript** – Paste text manually and click to extract topics and action items

## Tech Stack

- Python 3.9+
- Flask, Flask-SocketIO
- PyTorch, Transformers (Wav2Vec2)
- NLTK, spaCy, scikit-learn, Gensim
