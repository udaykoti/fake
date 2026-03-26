# Fake Job Detection System

An AI-powered system that analyzes job postings and returns a scam probability score using NLP, behavioral rules, domain intelligence, and company validation.

## Features

- **NLP Model** — TF-IDF + Logistic Regression baseline; DistilBERT fine-tuned classifier
- **Behavioral Detection** — 8 rule categories: payment requests, urgency language, WhatsApp/Telegram contact, no-interview claims, personal info requests, vague company, WFH bait, grammar red flags
- **Domain Intelligence** — WHOIS domain age, SSL validity, suspicious TLD/pattern detection
- **Company Validation** — known employer database lookup, industry-description consistency check
- **OCR** — Tesseract-powered text extraction from job posting screenshots
- **Scoring Engine** — weighted combination of all modules with human-readable explanation
- **REST API** — FastAPI backend, single `POST /analyze` endpoint
- **React UI** — clean frontend with text input, image upload, score gauge, and flag breakdown

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| NLP | scikit-learn, HuggingFace Transformers (DistilBERT), NLTK |
| OCR | Tesseract, pytesseract, Pillow |
| Domain | python-whois, ssl (stdlib) |
| Database | MongoDB (motor async driver) |
| Frontend | React, Vite |
| ML Ops | joblib, pandas, imbalanced-learn |

## Project Structure

```
ai-project/
├── backend/
│   ├── app/
│   │   ├── main.py                        # FastAPI app + lifespan
│   │   ├── routes/analyze.py              # POST /analyze endpoint
│   │   ├── services/
│   │   │   ├── preprocessor.py            # text cleaning pipeline
│   │   │   ├── risk_scorer.py             # unified NLP scorer
│   │   │   ├── behavioral_detector.py     # rule-based scam patterns
│   │   │   ├── domain_analyzer.py         # WHOIS + SSL + TLD checks
│   │   │   ├── company_validator.py       # company DB lookup
│   │   │   ├── scoring_engine.py          # weighted final score
│   │   │   └── ocr_service.py             # Tesseract OCR
│   │   ├── models/
│   │   │   ├── tfidf_logreg.py            # baseline ML model
│   │   │   └── distilbert_classifier.py   # transformer model + risk scorer
│   │   └── data/
│   │       └── prepare_dataset.py         # preprocessing + train/val/test split
│   ├── tests/test_pipeline.py             # 10-case integration test
│   └── requirements.txt
└── frontend/
    └── src/
        ├── App.jsx                        # main UI
        ├── api/analyze.js                 # API client
        └── components/
            ├── ScoreGauge.jsx             # visual score bar
            └── ResultPanel.jsx            # full result display
```

## System Architecture

```
User Input (text / image / URL)
         │
         ▼
   ┌─────────────┐
   │  FastAPI     │  POST /analyze
   │  Backend     │
   └──────┬──────┘
          │
    ┌─────┴──────────────────────────────┐
    │                                    │
    ▼                                    ▼
┌────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐
│  OCR   │  │   NLP    │  │ Behavior │  │ Domain  │
│Tesseract│  │DistilBERT│  │  Rules   │  │  WHOIS  │
└────┬───┘  └────┬─────┘  └────┬─────┘  └────┬────┘
     │           │              │              │
     └───────────┴──────────────┴──────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  Scoring Engine │  weighted combination
                 │  + Explanation  │
                 └────────┬────────┘
                          │
                          ▼
              { score, risk_level, flags,
                explanation, breakdown }
```

## Scoring Weights

| Module | Weight |
|---|---|
| NLP (text model) | 35% |
| Behavioral rules | 30% |
| Domain analysis | 20% |
| Company validation | 15% |

## Quick Start

### Backend
```bash
cd backend
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Train models
```bash
# Baseline
python -m app.models.tfidf_logreg --train data/processed/train.csv --val data/processed/val.csv --test data/processed/test.csv

# DistilBERT (GPU recommended)
python -m app.models.distilbert_classifier --train data/processed/train.csv --val data/processed/val.csv --test data/processed/test.csv
```

### Run tests
```bash
python tests/test_pipeline.py
```

## API Usage

```bash
# Text only
curl -X POST http://localhost:8000/analyze \
  -F "text=Earn $5000/week from home, no experience needed, send bank details" \
  -F "company_name=Global Solutions LLC"

# With URL
curl -X POST http://localhost:8000/analyze \
  -F "text=..." \
  -F "url=https://jobs123.xyz/apply-now"

# With image
curl -X POST http://localhost:8000/analyze \
  -F "image=@screenshot.png"
```

### Response
```json
{
  "final_score": 0.87,
  "risk_level": "CRITICAL",
  "explanation": "This job posting is highly likely to be fraudulent...",
  "breakdown": { "nlp": 0.94, "behavioral": 0.81, "domain": 0.75, "company": 0.90 },
  "flags": ["Asks candidate to pay money upfront", "Domain: very_new_domain (12d)"],
  "modules_used": ["nlp", "behavioral", "domain", "company"]
}
```

## Dataset

Primary: [Kaggle EMSCAD Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) — 17,880 labeled postings.

## Test Results (rule-based, no trained model)

10-case integration test using behavioral + company modules only:
- Real postings (5): correctly identified as low-risk
- Fake postings (5): correctly flagged as high/critical risk
- Accuracy: 10/10 (100% on synthetic test set)

*Full accuracy with trained NLP model on held-out test set: reported after training.*
