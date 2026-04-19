"""
model_persistence.py — Save and Load Trained Models
====================================================

PURPOSE:
    After training (which takes time), we save the trained model, TF-IDF
    vectorizer, and label encoder to disk as .pkl (pickle) files. Later,
    when predicting, we load them instead of retraining from scratch.

WHAT IS PICKLE?
    Pickle is Python's built-in serialization format. It converts Python
    objects (models, vectorizers, etc.) into binary files that can be
    saved to disk and loaded back later with the exact same state.

    Think of it like "freezing" a trained model and "thawing" it later.

HOW THE ORIGINAL NOTEBOOK DID IT:
    - Cell 15 (commented out):
        pickle.dump(tfidf_vectorizer, open('tfidf.pkl', 'wb'))
        pickle.dump(rf_classifier, open('rf_classifier.pkl', 'wb'))

    We improve this by:
    1. Saving to a dedicated models/ directory
    2. Also saving the label encoder (needed to decode predictions)
    3. Using proper file handling (with statement)
    4. Adding error handling and logging

USAGE:
    from src.model_persistence import save_artifacts, load_artifacts
    
    save_artifacts(model, tfidf, label_encoder)
    model, tfidf, label_encoder = load_artifacts()
"""

import os
import pickle

from src.config import (
    MODELS_DIR,
    TFIDF_MODEL_FILE,
    LABEL_ENCODER_FILE,
    BEST_MODEL_FILE,
)


def save_artifacts(model, tfidf_vectorizer, label_encoder) -> None:
    """
    Save trained model, TF-IDF vectorizer, and label encoder to disk.

    Args:
        model:            The trained classifier (e.g., LogisticRegression)
        tfidf_vectorizer: The fitted TfidfVectorizer
        label_encoder:    The fitted LabelEncoder

    Files created in models/ directory:
        - best_model.pkl        — the trained classifier
        - tfidf_vectorizer.pkl  — the fitted TF-IDF vectorizer
        - label_encoder.pkl     — the fitted label encoder

    Why save all three?
        - model: does the actual prediction (number → number)
        - tfidf_vectorizer: needed to convert NEW resume text → numbers
        - label_encoder: needed to convert predicted number → category name
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save each artifact
    artifacts = {
        BEST_MODEL_FILE: model,
        TFIDF_MODEL_FILE: tfidf_vectorizer,
        LABEL_ENCODER_FILE: label_encoder,
    }

    print(f"\n💾 Saving model artifacts to: {MODELS_DIR}")

    for filename, artifact in artifacts.items():
        filepath = os.path.join(MODELS_DIR, filename)
        with open(filepath, "wb") as f:
            pickle.dump(artifact, f)
        print(f"   ✅ Saved: {filename}")

    print("💾 All artifacts saved successfully!")


def load_artifacts() -> tuple:
    """
    Load previously saved model, TF-IDF vectorizer, and label encoder.

    Returns:
        tuple: (model, tfidf_vectorizer, label_encoder)

    Raises:
        FileNotFoundError: If any of the model files don't exist
            (meaning you haven't trained yet — run main.py first)
    """
    print(f"📂 Loading model artifacts from: {MODELS_DIR}")

    files = {
        "model": os.path.join(MODELS_DIR, BEST_MODEL_FILE),
        "tfidf": os.path.join(MODELS_DIR, TFIDF_MODEL_FILE),
        "label_encoder": os.path.join(MODELS_DIR, LABEL_ENCODER_FILE),
    }

    # Check all files exist before loading
    for name, filepath in files.items():
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"❌ Model file not found: {filepath}\n"
                f"   Have you trained the model? Run: python main.py"
            )

    # Load each artifact
    with open(files["model"], "rb") as f:
        model = pickle.load(f)
    print(f"   ✅ Loaded: {BEST_MODEL_FILE}")

    with open(files["tfidf"], "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    print(f"   ✅ Loaded: {TFIDF_MODEL_FILE}")

    with open(files["label_encoder"], "rb") as f:
        label_encoder = pickle.load(f)
    print(f"   ✅ Loaded: {LABEL_ENCODER_FILE}")

    print("📂 All artifacts loaded successfully!")
    return model, tfidf_vectorizer, label_encoder
