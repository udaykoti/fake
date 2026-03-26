"""
FastAPI entry point — mounts all routes and initializes shared services.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.routes.analyze import router as analyze_router
from app.services.risk_scorer import RiskScorerService

load_dotenv()

# ---------------------------------------------------------------------------
# Shared model instance (loaded once at startup)
# ---------------------------------------------------------------------------

_scorer: RiskScorerService = None


def get_scorer() -> RiskScorerService:
    return _scorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scorer
    distilbert_dir = os.getenv("DISTILBERT_MODEL_DIR", "")
    logreg_path    = os.getenv("LOGREG_MODEL_PATH", "models/saved/tfidf_logreg.pkl")

    try:
        _scorer = RiskScorerService(
            distilbert_dir=distilbert_dir or None,
            logreg_path=logreg_path,
        )
        print("[startup] Model loaded successfully.")
    except Exception as e:
        print(f"[startup] No trained model found ({e}). NLP scoring disabled — other modules still active.")
        _scorer = None

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fake Job Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _scorer is not None}
