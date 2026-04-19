"""
feature_engineering.py — Label Encoding + TF-IDF Vectorization
==============================================================

PURPOSE:
    ML models work with NUMBERS, not text. This module converts:
    1. Category labels (text) → numbers  (using LabelEncoder)
    2. Resume text → numerical feature vectors (using TF-IDF)
    3. Splits data into training and testing sets

WHAT IS LABEL ENCODING?
    "Data Science" → 6, "HR" → 12, "Testing" → 23, etc.
    LabelEncoder assigns each unique category a number.

WHAT IS TF-IDF?
    Measures how important a word is in a document vs all documents.
    TF = word frequency in one doc. IDF = how rare it is across all docs.
    TF-IDF = TF × IDF. High score = important distinguishing word.

USAGE:
    from src.feature_engineering import encode_labels, vectorize_text, split_data
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORY_COLUMN,
    CLEANED_RESUME_COLUMN,
    ENCODED_CATEGORY_COLUMN,
    TFIDF_MAX_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
)


def encode_labels(df: pd.DataFrame) -> tuple:
    """
    Convert text category labels to numbers using LabelEncoder.

    Args:
        df: DataFrame with a 'Category' column.

    Returns:
        tuple: (modified_df, label_encoder)
    """
    print("🏷️  Encoding category labels...")
    le = LabelEncoder()
    df[ENCODED_CATEGORY_COLUMN] = le.fit_transform(df[CATEGORY_COLUMN])

    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"✅ Encoded {len(mapping)} categories:")
    for name, code in mapping.items():
        print(f"   {code:2d} → {name}")

    return df, le


def vectorize_text(df: pd.DataFrame) -> tuple:
    """
    Convert cleaned resume text into TF-IDF numerical features.

    Args:
        df: DataFrame with a 'Cleaned Resume' column.

    Returns:
        tuple: (X, tfidf_vectorizer)
    """
    print(f"📊 Vectorizing text with TF-IDF (max_features={TFIDF_MAX_FEATURES})...")
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    X = tfidf.fit_transform(df[CLEANED_RESUME_COLUMN])
    print(f"✅ TF-IDF matrix shape: {X.shape}")
    print(f"   ({X.shape[0]} resumes × {X.shape[1]} features)")
    return X, tfidf


def split_data(X, y) -> tuple:
    """
    Split data into training (80%) and testing (20%) sets.

    Args:
        X: Feature matrix (TF-IDF vectors)
        y: Target labels (encoded categories)

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    print(f"✂️  Splitting data: {1-TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test...")
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    print(f"✅ Training set: {x_train.shape[0]} samples")
    print(f"✅ Testing set:  {x_test.shape[0]} samples")
    return x_train, x_test, y_train, y_test


# ============================================================
# OPTIONAL: Oversampling / Balancing (from commented-out notebook code)
# Uncomment if you want to balance training data in the future.
# ============================================================
# from sklearn.utils import resample
#
# def balance_dataset(df, label_column):
#     max_count = df[label_column].value_counts().max()
#     balanced_data = []
#     for category in df[label_column].unique():
#         category_data = df[df[label_column] == category]
#         if len(category_data) < max_count:
#             balanced = resample(category_data, replace=True,
#                                 n_samples=max_count, random_state=42)
#         else:
#             balanced = resample(category_data, replace=False,
#                                 n_samples=max_count, random_state=42)
#         balanced_data.append(balanced)
#     return pd.concat(balanced_data)
