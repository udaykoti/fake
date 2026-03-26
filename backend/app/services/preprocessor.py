"""
Job Posting NLP Preprocessor
Handles cleaning, tokenization, imbalance, and train/test splitting.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Download required NLTK data on first run
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# 1. TEXT CLEANING
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Full cleaning pipeline for a single text field."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove phone numbers
    text = re.sub(r"\+?\d[\d\s\-().]{7,}\d", " ", text)

    # Remove special characters and digits (keep letters and spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_noise(text: str) -> str:
    """Remove stopwords and very short tokens."""
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def lemmatize_text(text: str) -> str:
    """Lemmatize tokens to their base form."""
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def full_clean(text: str) -> str:
    """Apply all cleaning steps in sequence."""
    return lemmatize_text(remove_noise(clean_text(text)))


# ---------------------------------------------------------------------------
# 2. TOKENIZATION
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Word tokenization using NLTK."""
    return word_tokenize(text)


def tokenize_series(series: pd.Series) -> pd.Series:
    """Tokenize a full pandas Series, returns list of tokens per row."""
    return series.apply(tokenize)


# ---------------------------------------------------------------------------
# 3. FEATURE COMBINATION
# ---------------------------------------------------------------------------

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]


def combine_text_fields(df: pd.DataFrame) -> pd.Series:
    """
    Merge all relevant text columns into one field.
    Missing fields are treated as empty strings.
    """
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    combined = df[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)
    return combined


# ---------------------------------------------------------------------------
# 4. FULL PREPROCESSING PIPELINE
# ---------------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end preprocessing on a raw job postings DataFrame.
    Expects a 'fraudulent' column as the label.
    """
    df = df.copy()

    print(f"[preprocess] Raw shape: {df.shape}")
    print(f"[preprocess] Label distribution:\n{df['fraudulent'].value_counts()}")

    # Combine text fields
    df["combined_text"] = combine_text_fields(df)

    # Clean
    df["clean_text"] = df["combined_text"].apply(full_clean)

    # Drop rows where cleaning left nothing
    df = df[df["clean_text"].str.strip() != ""]

    # Tokenized column (useful for embedding models)
    df["tokens"] = tokenize_series(df["clean_text"])

    print(f"[preprocess] After cleaning: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 5. HANDLING CLASS IMBALANCE
# ---------------------------------------------------------------------------

def handle_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "smote"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance classes using SMOTE (oversample minority) or undersample majority.

    strategy: 'smote' | 'undersample' | 'combined'
    """
    print(f"[imbalance] Before — class counts: {np.bincount(y)}")

    if strategy == "smote":
        sampler = SMOTE(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

    elif strategy == "combined":
        # Oversample minority to 50% of majority, then undersample majority
        pipeline = ImbPipeline([
            ("over", SMOTE(sampling_strategy=0.5, random_state=42)),
            ("under", RandomUnderSampler(sampling_strategy=0.8, random_state=42)),
        ])
        X_res, y_res = pipeline.fit_resample(X, y)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"[imbalance] After  — class counts: {np.bincount(y_res)}")
    return X_res, y_res


# ---------------------------------------------------------------------------
# 6. TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    label_col: str = "fraudulent",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into train / validation / test sets.
    Default: 70% train, 15% val, 15% test.
    """
    # First split off test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state,
    )

    # Split remaining into train and val
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[label_col],
        random_state=random_state,
    )

    print(f"[split] Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        dist = split[label_col].value_counts(normalize=True).round(3).to_dict()
        print(f"  {name} label dist: {dist}")

    return train, val, test
