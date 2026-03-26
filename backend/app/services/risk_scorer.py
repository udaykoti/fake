"""
Unified risk scoring service.
Tries DistilBERT first, falls back to TF-IDF + LogReg if not available.
This is what FastAPI routes will call.
"""

import os
from app.services.preprocessor import full_clean


class RiskScorerService:
    """
    Wraps either DistilBERT or TF-IDF model behind a single interface.
    Instantiate once at app startup and reuse.
    """

    def __init__(self, distilbert_dir: str = None, logreg_path: str = None):
        self.model_type = None
        self._scorer    = None

        if distilbert_dir and os.path.isdir(distilbert_dir):
            from app.models.distilbert_classifier import FakeJobRiskScorer
            self._scorer    = FakeJobRiskScorer(distilbert_dir)
            self.model_type = "distilbert"
            print(f"[RiskScorer] Loaded DistilBERT from {distilbert_dir}")

        elif logreg_path and os.path.isfile(logreg_path):
            import joblib
            self._model     = joblib.load(logreg_path)
            self.model_type = "logreg"
            print(f"[RiskScorer] Loaded TF-IDF+LogReg from {logreg_path}")

        else:
            raise FileNotFoundError(
                "No model found. Provide distilbert_dir or logreg_path."
            )

    def score(self, text: str) -> dict:
        """
        Score a raw job posting text (cleaning applied internally).

        Returns:
            {
                "risk_score": float,   # 0.0–1.0
                "label":      str,     # "FAKE" | "REAL"
                "confidence": float,
                "risk_level": str,     # "LOW" | "MEDIUM" | "HIGH"
                "model":      str,
            }
        """
        clean = full_clean(text)

        if self.model_type == "distilbert":
            result = self._scorer.score(clean)
        else:
            result = self._logreg_score(clean)

        result["model"] = self.model_type
        return result

    def score_fields(
        self,
        title: str = "",
        company: str = "",
        description: str = "",
        requirements: str = "",
        benefits: str = "",
    ) -> dict:
        """
        Score from individual job posting fields.
        Mirrors the structure of the training data.
        """
        combined = " ".join([title, company, description, requirements, benefits])
        return self.score(combined)

    def _logreg_score(self, clean_text: str) -> dict:
        from app.models.tfidf_logreg import _risk_level  # reuse helper
        prob  = self._model.predict_proba([clean_text])[0]
        label = "FAKE" if prob[1] >= 0.5 else "REAL"
        return {
            "risk_score": round(float(prob[1]), 4),
            "label":      label,
            "confidence": round(float(prob[1] if label == "FAKE" else prob[0]), 4),
            "risk_level": _risk_level(prob[1]),
        }
