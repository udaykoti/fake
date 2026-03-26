"""
Smoke test — runs the full pipeline on synthetic data.
No real dataset needed.

Run:
    python -m app.models.test_model_smoke
"""

import pandas as pd
import numpy as np
from app.services.preprocessor import full_clean
from app.models.tfidf_logreg import build_pipeline, evaluate, predict

# --- Generate tiny synthetic dataset ---
np.random.seed(42)

REAL_DESCS = [
    "We are looking for a software engineer with 3 years of Python experience.",
    "Join our team as a data analyst. Strong SQL and Excel skills required.",
    "Marketing manager needed. MBA preferred. Competitive salary and benefits.",
    "Full stack developer role. React and Node.js experience essential.",
    "Customer support specialist. Excellent communication skills required.",
]

FAKE_DESCS = [
    "Earn $5000 per week working from home! No experience needed. Send bank details.",
    "Immediate hiring! No skills required. Guaranteed income. Work anywhere.",
    "Make money fast. Data entry job. $500/day. No interview needed.",
    "Work from home opportunity. Earn thousands weekly. No qualifications needed.",
    "Urgent hiring! Send your SSN and bank account to receive your starter kit.",
]

rows = []
for desc in REAL_DESCS * 20:
    rows.append({"clean_text": full_clean(desc), "fraudulent": 0})
for desc in FAKE_DESCS * 20:
    rows.append({"clean_text": full_clean(desc), "fraudulent": 1})

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

split = int(len(df) * 0.8)
train_df, test_df = df[:split], df[split:]

# --- Train ---
print("Training on synthetic data...")
model = build_pipeline()
model.fit(train_df["clean_text"], train_df["fraudulent"])

# --- Evaluate ---
evaluate(model, test_df["clean_text"], test_df["fraudulent"], "Smoke Test")

# --- Single prediction ---
print("\n--- Single prediction examples ---")
samples = [
    "Hiring Python developer, 5 years experience, competitive salary",
    "Work from home earn 5000 weekly no experience send bank details",
]
for s in samples:
    result = predict(model, full_clean(s))
    label = "FAKE" if result["fraudulent"] else "REAL"
    print(f"  [{label}] conf={result['confidence']} | '{s[:60]}'")

print("\nSmoke test passed.")
