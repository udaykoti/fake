"""
Fake Job Detection — TF-IDF + Logistic Regression
Baseline model: fast to train, interpretable, strong on text classification.

Usage:
    python -m app.models.tfidf_logreg --train data/processed/train.csv \
                                       --val   data/processed/val.csv  \
                                       --test  data/processed/test.csv \
                                       --out   models/saved/
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    TF-IDF (1+2-grams) → Logistic Regression pipeline.
    class_weight='balanced' handles imbalance without needing SMOTE
    when working directly from raw text (no pre-vectorized arrays).
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=15_000,
            ngram_range=(1, 2),   # unigrams + bigrams
            min_df=2,             # ignore terms appearing in < 2 docs
            sublinear_tf=True,    # apply log(1+tf) — helps with long docs
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=1.0,                # regularization strength (tune if needed)
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: Pipeline, X: pd.Series, y: pd.Series, split_name: str) -> dict:
    """Print and return a full evaluation report."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "split":     split_name,
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y, y_prob), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {split_name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  Precision : {metrics['precision']}  (of flagged jobs, how many are truly fake)")
    print(f"  Recall    : {metrics['recall']}  (of all fake jobs, how many did we catch)")
    print(f"  F1 Score  : {metrics['f1']}")
    print(f"  ROC-AUC   : {metrics['roc_auc']}")
    print(f"\nClassification Report:\n{classification_report(y, y_pred, target_names=['Real','Fake'])}")

    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")

    return metrics


# ---------------------------------------------------------------------------
# Top features — interpretability
# ---------------------------------------------------------------------------

def show_top_features(model: Pipeline, n: int = 20):
    """Print the words most associated with fake vs real postings."""
    tfidf = model.named_steps["tfidf"]
    clf   = model.named_steps["clf"]
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]

    top_fake = feature_names[np.argsort(coefs)[-n:][::-1]]
    top_real = feature_names[np.argsort(coefs)[:n]]

    print(f"\nTop {n} words → FAKE: {list(top_fake)}")
    print(f"Top {n} words → REAL: {list(top_real)}")


# ---------------------------------------------------------------------------
# Train + save
# ---------------------------------------------------------------------------

def train(train_path: str, val_path: str, test_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Load splits
    train_df = pd.read_csv(train_path).fillna("")
    val_df   = pd.read_csv(val_path).fillna("")
    test_df  = pd.read_csv(test_path).fillna("")

    X_train, y_train = train_df["clean_text"], train_df["fraudulent"]
    X_val,   y_val   = val_df["clean_text"],   val_df["fraudulent"]
    X_test,  y_test  = test_df["clean_text"],  test_df["fraudulent"]

    # Build and train
    print("Training TF-IDF + Logistic Regression...")
    model = build_pipeline()
    model.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate on all splits
    train_metrics = evaluate(model, X_train, y_train, "Train")
    val_metrics   = evaluate(model, X_val,   y_val,   "Validation")
    test_metrics  = evaluate(model, X_test,  y_test,  "Test")

    # Interpretability
    show_top_features(model)

    # Save model
    model_path = os.path.join(out_dir, "tfidf_logreg.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved → {model_path}")

    # Save metrics summary
    metrics_df = pd.DataFrame([train_metrics, val_metrics, test_metrics])
    metrics_path = os.path.join(out_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved → {metrics_path}")

    return model


# ---------------------------------------------------------------------------
# Inference helper (used by FastAPI later)
# ---------------------------------------------------------------------------

def load_model(model_path: str) -> Pipeline:
    return joblib.load(model_path)


def _risk_level(score: float) -> str:
    if score < 0.35:
        return "LOW"
    elif score < 0.65:
        return "MEDIUM"
    else:
        return "HIGH"


def predict(model: Pipeline, text: str) -> dict:
    """
    Predict a single job posting.
    Returns label + confidence score.
    """
    prob = model.predict_proba([text])[0]
    label = int(model.predict([text])[0])
    return {
        "fraudulent": label,
        "confidence": round(float(prob[label]), 4),
        "prob_fake":  round(float(prob[1]), 4),
        "prob_real":  round(float(prob[0]), 4),
        "risk_level": _risk_level(float(prob[1])),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val",   default="data/processed/val.csv")
    parser.add_argument("--test",  default="data/processed/test.csv")
    parser.add_argument("--out",   default="models/saved")
    args = parser.parse_args()

    train(args.train, args.val, args.test, args.out)
