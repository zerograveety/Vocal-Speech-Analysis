# Vocal Check

**AI-powered presentation coach** — upload a video or slide deck and get instant, actionable feedback on your speaking and delivery.

Vocal Check analyzes **speech pace, pauses, filler words, pronunciation, body language, eye contact, and slide design**, then uses Google Gemini to turn the raw metrics into clear, human feedback you can act on. Built for students preparing for viva/presentations, professionals refining public speaking, and trainers giving structured coaching.

---

## Features

- 🎤 **Voice & speech analysis** — pacing, pauses, filler words, pronunciation, clarity, and tone.
- 🎥 **Body-language tracking** — posture, eye contact, and gesture via MediaPipe pose/hand detection (with an OpenCV fallback).
- 📊 **Slide-deck analysis** — text density and visual balance for `.pptx` uploads, plus optional grammar quality via a BERT-based CoLA classifier.
- 🧠 **AI-powered insights** — Gemini synthesizes the raw metrics into clear, actionable feedback and an overall score.
- 📄 **Detailed reports** — a results page summarizing every metric with recommendations.
- 🛡️ **Graceful degradation** — optional heavy dependencies (torch/transformers, mediapipe) are detected at runtime and skipped if missing.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python · Flask |
| AI / ML | Google Gemini API · MediaPipe · OpenCV · transformers (optional) |
| Frontend | HTML · CSS · Vanilla JavaScript |
| File handling | Werkzeug (secure uploads) |

---

## Getting Started

### Prerequisites

- Python 3.8+
- A Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### Installation

```bash
git clone https://github.com/zerograveety/Vocal-Speech-Analysis.git
cd Vocal-Speech-Analysis
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Set your API key as an environment variable (do **not** hardcode it or commit `.env`):

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=your_key_here
export $(grep -v '^#' .env | xargs)   # or use a dotenv loader
```

### Run

```bash
python app.py
```

Then open http://localhost:5000, and upload a video (`.mp4`/`.mov`) or slide deck (`.pptx`).

---

## Project Structure

```
app.py              # Flask application & routes
video_analysis.py   # MediaPipe/OpenCV body + speech analysis
ppt_analysis.py     # PPTX parsing, text density, grammar (CoLA)
gemini_analysis.py  # Turns raw metrics into Gemini feedback
templates/          # HTML templates (index, landing, practice, ppt, results)
static/             # CSS and images
test_api.py         # API tests
test_installation.py# Environment smoke test
```

---

## Notes

- Videos up to 16MB are supported; processing time depends on length and complexity.
- AI features require a working internet connection and a valid `GEMINI_API_KEY`.
- `torch`/`transformers` are optional — install them if you want the grammar classifier (see `requirements.txt`).
