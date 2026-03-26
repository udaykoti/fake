"""
Fake Job Detection — DistilBERT Fine-tuning
Prompt 4: Training loop + evaluation
Prompt 5: Risk scoring function

Usage:
    python -m app.models.distilbert_classifier --train data/processed/train.csv \
                                                --val   data/processed/val.csv  \
                                                --test  data/processed/test.csv \
                                                --out   models/saved/distilbert
"""

import argparse
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model_name:     str   = "distilbert-base-uncased"
    max_length:     int   = 256      # job descriptions are long; 256 is a good balance
    batch_size:     int   = 16
    epochs:         int   = 4
    learning_rate:  float = 2e-5
    warmup_ratio:   float = 0.1
    weight_decay:   float = 0.01
    grad_clip:      float = 1.0
    seed:           int   = 42
    device:         str   = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JobPostingDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def load_split(path: str) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path).fillna("")
    # Use clean_text if available, fall back to raw description
    text_col = "clean_text" if "clean_text" in df.columns else "description"
    return df[text_col].tolist(), df["fraudulent"].astype(int).tolist()


# ---------------------------------------------------------------------------
# Class weights for imbalanced data
# ---------------------------------------------------------------------------

def compute_class_weights(labels: list[int], device: str) -> torch.Tensor:
    counts = np.bincount(labels)
    total  = len(labels)
    weights = total / (len(counts) * counts)   # inverse frequency
    return torch.tensor(weights, dtype=torch.float).to(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, cfg: TrainConfig) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)
        labels         = batch["labels"].to(cfg.device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, loader, cfg: TrainConfig, split_name: str) -> dict:
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            labels         = batch["labels"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds   = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_probs.extend(probs[:, 1])   # prob of class 1 (fake)
            all_labels.extend(labels)

    metrics = {
        "split":     split_name,
        "accuracy":  round(accuracy_score(all_labels, all_preds), 4),
        "precision": round(precision_score(all_labels, all_preds, zero_division=0), 4),
        "recall":    round(recall_score(all_labels, all_preds, zero_division=0), 4),
        "f1":        round(f1_score(all_labels, all_preds, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(all_labels, all_probs), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {split_name} Results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "split":
            print(f"  {k:<12}: {v}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['Real', 'Fake'])}")

    return metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(train_path: str, val_path: str, test_path: str, out_dir: str):
    cfg = TrainConfig()
    torch.manual_seed(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device: {cfg.device}")

    # Load data
    X_train, y_train = load_split(train_path)
    X_val,   y_val   = load_split(val_path)
    X_test,  y_test  = load_split(test_path)

    # Tokenizer
    print(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg.model_name)

    # Datasets
    train_ds = JobPostingDataset(X_train, y_train, tokenizer, cfg.max_length)
    val_ds   = JobPostingDataset(X_val,   y_val,   tokenizer, cfg.max_length)
    test_ds  = JobPostingDataset(X_test,  y_test,  tokenizer, cfg.max_length)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size)

    # Model
    print(f"Loading model: {cfg.model_name}")
    model = DistilBertForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2
    ).to(cfg.device)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps   = len(train_loader) * cfg.epochs
    warmup_steps  = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop with early stopping on val F1
    best_val_f1   = 0.0
    best_model_path = os.path.join(out_dir, "best_model")

    for epoch in range(1, cfg.epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, cfg)
        print(f"\nEpoch {epoch}/{cfg.epochs} — avg train loss: {avg_loss:.4f}")

        val_metrics = evaluate_model(model, val_loader, cfg, f"Validation (epoch {epoch})")

        # Save best checkpoint
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  ✓ New best model saved (val F1={best_val_f1})")

    # Final evaluation on test set using best checkpoint
    print("\nLoading best checkpoint for test evaluation...")
    best_model = DistilBertForSequenceClassification.from_pretrained(best_model_path).to(cfg.device)
    test_metrics = evaluate_model(best_model, test_loader, cfg, "Test (best checkpoint)")

    # Save metrics
    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv(os.path.join(out_dir, "distilbert_metrics.csv"), index=False)
    print(f"\nDone. Best model saved to {best_model_path}")


# ---------------------------------------------------------------------------
# PROMPT 5 — Risk Scorer
# ---------------------------------------------------------------------------

class FakeJobRiskScorer:
    """
    Loads a fine-tuned DistilBERT model and scores job postings.
    Returns a risk score between 0.0 (definitely real) and 1.0 (definitely fake).
    """

    def __init__(self, model_dir: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def score(self, text: str) -> dict:
        """
        Score a single job posting text.

        Returns:
            {
                "risk_score": float,   # 0.0–1.0, probability of being fake
                "label": str,          # "FAKE" | "REAL"
                "confidence": float,   # confidence in the predicted label
                "risk_level": str,     # "LOW" | "MEDIUM" | "HIGH"
            }
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        risk_score = float(probs[1])   # probability of class 1 (fake)
        label      = "FAKE" if risk_score >= 0.5 else "REAL"
        confidence = float(probs[1] if label == "FAKE" else probs[0])

        return {
            "risk_score": round(risk_score, 4),
            "label":      label,
            "confidence": round(confidence, 4),
            "risk_level": _risk_level(risk_score),
        }

    def score_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Score multiple job postings efficiently."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()

            for prob in probs:
                risk_score = float(prob[1])
                label      = "FAKE" if risk_score >= 0.5 else "REAL"
                confidence = float(prob[1] if label == "FAKE" else prob[0])
                results.append({
                    "risk_score": round(risk_score, 4),
                    "label":      label,
                    "confidence": round(confidence, 4),
                    "risk_level": _risk_level(risk_score),
                })
        return results


def _risk_level(score: float) -> str:
    if score < 0.35:
        return "LOW"
    elif score < 0.65:
        return "MEDIUM"
    else:
        return "HIGH"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val",   default="data/processed/val.csv")
    parser.add_argument("--test",  default="data/processed/test.csv")
    parser.add_argument("--out",   default="models/saved/distilbert")
    args = parser.parse_args()

    train(args.train, args.val, args.test, args.out)
