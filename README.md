# ORION — Oral Reasoning & Interview Optimization Network

A full-stack, multimodal AI system for practising interviews and vivas. A candidate records answers in the browser; ORION transcribes the speech and scores **content**, **fluency** and **non-verbal communication**, tracks progress across sessions, and gives coaching feedback.

ORION is a **practice and coaching tool**, not an automated hiring or grading system.

**Authors:** Anukriti Singh, Rishita Sinha, G. Renuka Reddy — School of Computer Science Engg. & Tech., Bennett University

---

## Features

- Browser-based recording (4 random questions per session, review/retake before submitting)
- Speech-to-text with Whisper
- Content scoring: semantic relevance (Sentence-BERT), answer structure, vocabulary, length, plus an optional custom-trained classifier
- Fluency scoring: words per minute, filler words, pauses, pitch variation (librosa + Praat)
- Non-verbal scoring: eye contact, posture and face presence (MediaPipe) and vocal emotion (audio model trained on RAVDESS)
- Authenticity check: compares audible speech with visible mouth movement
- Score history, improvement deltas against the previous session, and rule-based coaching tips

---

## Project Structure

```
orion-ai-interview-system/
├── backend/                         # FastAPI backend
│   ├── main.py                      # App entry point
│   ├── database.py                  # PostgreSQL config (SQLAlchemy)
│   ├── models.py                    # ORM models
│   ├── requirements.txt
│   ├── .env.example                 # Environment variable template
│   ├── alembic/                     # Database migrations
│   ├── routers/
│   │   ├── auth.py                  # Register, login, JWT
│   │   ├── sessions.py              # Create session, upload answers, complete
│   │   ├── analysis.py              # Analysis results + score history
│   │   └── questions.py             # Question bank
│   ├── services/
│   │   ├── pipeline.py              # Orchestrates all analysis steps
│   │   ├── audio_extractor.py       # ffmpeg: video -> 16 kHz mono WAV
│   │   ├── transcriber.py           # Whisper speech-to-text
│   │   ├── speech_analyser.py       # librosa + Praat: fluency metrics
│   │   ├── content_analyser.py      # SBERT relevance, structure, vocab, length
│   │   ├── emotion_analyser.py      # Audio emotion inference (RAVDESS model)
│   │   ├── video_analyser.py        # MediaPipe: eye contact, posture
│   │   └── feedback_generator.py    # Rule-based coaching feedback
│   └── ml/
│       ├── train_custom_model.py    # Trains the content classifier
│       ├── train_emotion_model.py   # Trains the RAVDESS emotion model
│       ├── dataset/                 # (not in git) weak/ average/ strong/ audio
│       └── ravdess/                 # (not in git) extracted RAVDESS speech files
└── frontend/                        # React + Vite frontend
    └── src/
        ├── App.jsx
        ├── api/client.js
        ├── hooks/useAuth.jsx
        ├── components/Layout.jsx
        └── pages/
            ├── LoginPage.jsx
            ├── RegisterPage.jsx
            ├── DashboardPage.jsx    # Score history + trend chart
            ├── InterviewPage.jsx    # Webcam recording flow
            └── AnalysisPage.jsx     # Full analysis results
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- ffmpeg (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Ubuntu)

### 1. Database

```bash
psql -U postgres
CREATE DATABASE orion_db;
\q
```

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set DATABASE_URL and SECRET_KEY

python -c "from database import engine, Base; import models; Base.metadata.create_all(engine)"
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

App: http://localhost:3000

---

## Training the Emotion Model (RAVDESS)

The emotion model is **not stored in the repository**, so you need to train it once. It is optional: without it, ORION still runs and uses a neutral default (50%) for the emotion term.

### 1. Download the dataset

1. Open https://zenodo.org/records/1188976
2. In the **Files** section, download **`Audio_Speech_Actors_01-24.zip`** (about 200 MB, 1,440 speech clips from 24 actors). The song and video files are not needed.

### 2. Place the files

Extract the zip into `backend/ml/ravdess/` so you have:

```
backend/ml/ravdess/Actor_01/...wav
backend/ml/ravdess/Actor_02/...wav
...
backend/ml/ravdess/Actor_24/...wav
```

### 3. Train

```bash
cd backend
source venv/bin/activate
python ml/train_emotion_model.py
```

This extracts audio features (MFCC, chroma and others), trains an SVM, prints held-out accuracy and a per-emotion report, and saves `ml/emotion_model.joblib`. Training takes a few minutes on a laptop CPU. To use less data: `python ml/train_emotion_model.py --actors 12`.

The train/test split is done **by actor**, so the same speaker never appears in both sets.

### 4. Enable it

Add to `backend/.env` (this is also the default path):

```
EMOTION_MODEL_PATH=ml/emotion_model.joblib
```

RAVDESS contains acted speech, so the emotion output is treated as a supporting signal, not a measure of interview quality.

---

## Training the Custom Content Model (optional)

Organise your labelled audio recordings:

```
backend/ml/dataset/weak/       # weak answers (.wav)
backend/ml/dataset/average/    # average answers (.wav)
backend/ml/dataset/strong/     # strong answers (.wav)
```

```bash
cd backend
python ml/train_custom_model.py
```

Then set `CUSTOM_MODEL_PATH=ml/orion_model.joblib` in `.env`. When loaded, the classifier contributes 25% of the content score.

---

## How the Pipeline Works

When a session is completed, `services/pipeline.py` processes each of the four answers:

```
Video file
   │
   ├─► audio_extractor.py  → 16 kHz mono WAV (ffmpeg)
   ├─► transcriber.py      → transcript + duration (Whisper)
   ├─► emotion_analyser.py → dominant emotion + positive-tone % (RAVDESS model)
   ├─► speech_analyser.py  → WPM, fillers, pauses, pitch (librosa, Praat)
   ├─► content_analyser.py → relevance, structure, vocab, length (SBERT)
   └─► video_analyser.py   → eye contact, posture, face presence, mouth movement (MediaPipe)

Per-answer scores → session scores → overall score, deltas vs previous session,
coaching feedback → saved to PostgreSQL
```

---

## Scoring

**Content (C)**

```
weighted = 0.35·relevance + 0.25·structure + 0.20·vocabulary + 0.20·length
           (with the custom model: 0.25 / 0.20 / 0.15 / 0.15 + 0.25·classifier)
C = weighted × relevance_gate
```

- Relevance is the cosine similarity between question and answer, rescaled so unrelated answers score near 0.
- Structure counts distinct signposting phrases ("for example", "as a result", "in conclusion"...).
- Vocabulary is reduced for very short answers.
- `relevance_gate` scales the whole score down when the answer is off-topic.
- Hostile or refusing answers ("I won't answer this") are capped at a low score.

**Fluency (F)**

```
F = 0.40·WPM_score + 0.30·(100 − filler_penalty) + 0.20·pause_score + 0.10·pitch_variety
```

**Non-verbal (N)**

```
N = 0.40·eye_contact% + 0.30·posture% + 0.20·positive_emotion% + 0.10·face_presence%
```

`positive_emotion%` comes from the audio emotion model (calm, happy and neutral count as positive).

**Overall**

```
Overall = (0.40·C + 0.35·F + 0.25·N) × content_factor × authenticity_factor
content_factor = 0.4 + 0.6 · min(C / 60, 1)
```

- `content_factor` stops fluent delivery and good posture from hiding poor content. It has no effect once C reaches 60.
- `authenticity_factor` is the mean per-answer authenticity (audible speech and visible mouth movement both present = 1.0).

These scores are heuristic signals for practice. They measure topic match, structure, delivery and visible behaviour, not whether an answer is actually correct.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Recharts |
| Backend | FastAPI, SQLAlchemy, Alembic |
| Database | PostgreSQL |
| Auth | JWT (python-jose + passlib) |
| Transcription | OpenAI Whisper (local) |
| Speech analysis | librosa, Parselmouth (Praat) |
| Content analysis | sentence-transformers, scikit-learn |
| Video analysis | MediaPipe, OpenCV |
| Emotion model | SVM on audio features, trained on RAVDESS |
| Custom content model | GradientBoostingClassifier |

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | required |
| `SECRET_KEY` | JWT signing secret (change in production) | required |
| `WHISPER_MODEL` | Whisper size: tiny / base / small / medium | `base` |
| `CUSTOM_MODEL_PATH` | Path to the trained content model | empty (disabled) |
| `EMOTION_MODEL_PATH` | Path to the trained RAVDESS emotion model | `ml/emotion_model.joblib` |
| `UPLOAD_DIR` | Directory for uploaded videos | `uploads` |

---

## Limitations

- The content classifier is trained on a small, project-collected dataset (75 recordings), so it is only a supplementary signal.
- RAVDESS is acted speech, not spontaneous interview speech.
- Results need further validation across speakers, accents and recording conditions before use outside a practice setting.

## Dataset Credit

Livingstone, S. R. and Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391. https://doi.org/10.5281/zenodo.1188976
