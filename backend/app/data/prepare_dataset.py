"""
Dataset preparation script.
Run this once to produce clean train/val/test CSVs.

Usage:
    python -m app.data.prepare_dataset --input data/raw_jobs.csv --output data/processed/
"""

import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.preprocessor import (
    preprocess_dataframe,
    split_dataset,
    handle_imbalance,
)


def main(input_path: str, output_dir: str, imbalance_strategy: str = "smote"):
    os.makedirs(output_dir, exist_ok=True)

    # --- Load ---
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    # Ensure required columns exist
    assert "fraudulent" in df.columns, "Dataset must have a 'fraudulent' column (0/1)"

    # --- Preprocess ---
    df = preprocess_dataframe(df)

    # --- Split first (prevents data leakage) ---
    train, val, test = split_dataset(df)

    # --- Vectorize (fit ONLY on train) ---
    print("[vectorize] Fitting TF-IDF on training set...")
    tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), min_df=2)
    X_train = tfidf.fit_transform(train["clean_text"])
    X_val   = tfidf.transform(val["clean_text"])
    X_test  = tfidf.transform(test["clean_text"])

    y_train = train["fraudulent"].values
    y_val   = val["fraudulent"].values
    y_test  = test["fraudulent"].values

    # --- Handle imbalance (train only) ---
    X_train_bal, y_train_bal = handle_imbalance(X_train, y_train, strategy=imbalance_strategy)

    # --- Save processed text splits ---
    train.to_csv(f"{output_dir}/train.csv", index=False)
    val.to_csv(f"{output_dir}/val.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)

    # Save vectorizer for inference
    import joblib
    joblib.dump(tfidf, f"{output_dir}/tfidf_vectorizer.pkl")

    print(f"\nDone. Files saved to {output_dir}/")
    print(f"  train.csv     : {len(train)} rows")
    print(f"  val.csv       : {len(val)} rows")
    print(f"  test.csv      : {len(test)} rows")
    print(f"  tfidf_vectorizer.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default="data/raw_jobs.csv")
    parser.add_argument("--output",   default="data/processed")
    parser.add_argument("--imbalance", default="smote", choices=["smote", "undersample", "combined"])
    args = parser.parse_args()

    main(args.input, args.output, args.imbalance)
