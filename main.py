"""
main.py — Training Entry Point
================================

PURPOSE:
    This is the MAIN script you run to train the Resume Analyzer.
    It executes the FULL pipeline from start to finish:

    1. Download dataset from Kaggle
    2. Clean resume text (preprocessing)
    3. Encode category labels (text → numbers)
    4. Vectorize text with TF-IDF (text → numerical features)
    5. Split into train/test sets
    6. Train multiple models and select the best
    7. Save trained models to disk

HOW TO RUN:
    python main.py

    After this completes, your trained models will be saved in the models/
    directory. You can then use predict.py to make predictions without
    retraining.

WHAT THIS REPLACES:
    All the active (non-commented) cells in Untitled15.ipynb, run in sequence.
"""

from src.data_loader import load_dataset
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import encode_labels, vectorize_text, split_data
from src.model_training import train_and_select_best_model
from src.model_persistence import save_artifacts
from src.config import ENCODED_CATEGORY_COLUMN


def main():
    """Execute the full training pipeline."""
    print("=" * 60)
    print("🚀 RESUME ANALYZER — TRAINING PIPELINE")
    print("=" * 60)

    # ---- Step 1: Load Data ----
    print("\n📦 STEP 1: Loading Dataset")
    print("-" * 40)
    df = load_dataset()

    # ---- Step 2: Preprocess Text ----
    print("\n🧹 STEP 2: Preprocessing Resume Text")
    print("-" * 40)
    df = preprocess_dataframe(df)

    # ---- Step 3: Encode Labels ----
    print("\n🏷️  STEP 3: Encoding Category Labels")
    print("-" * 40)
    df, label_encoder = encode_labels(df)

    # ---- Step 4: Vectorize Text ----
    print("\n📊 STEP 4: TF-IDF Vectorization")
    print("-" * 40)
    X, tfidf_vectorizer = vectorize_text(df)
    y = df[ENCODED_CATEGORY_COLUMN]

    # ---- Step 5: Split Data ----
    print("\n✂️  STEP 5: Train/Test Split")
    print("-" * 40)
    x_train, x_test, y_train, y_test = split_data(X, y)

    # ---- Step 6: Train Models ----
    print("\n🏋️  STEP 6: Training & Model Selection")
    print("-" * 40)
    best_model, best_name, best_acc, all_results = train_and_select_best_model(
        x_train, x_test, y_train, y_test
    )

    # ---- Step 7: Save Artifacts ----
    print("\n💾 STEP 7: Saving Trained Models")
    print("-" * 40)
    save_artifacts(best_model, tfidf_vectorizer, label_encoder)

    # ---- Done! ----
    print("\n" + "=" * 60)
    print(f"✅ TRAINING COMPLETE!")
    print(f"   Best Model: {best_name}")
    print(f"   Accuracy:   {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"\n   Next step: Run 'python predict.py' to make predictions!")
    print("=" * 60)


if __name__ == "__main__":
    main()
