"""
model_training.py — Train, Compare, and Select the Best ML Model
================================================================

PURPOSE:
    This module trains multiple ML classifiers on the resume data, compares
    their accuracy, and returns the best-performing model.

MODELS USED:
    1. Naive Bayes (MultinomialNB)     — Fast, good baseline for text classification
    2. Logistic Regression             — Strong linear classifier, often best for text
    3. SVM (Support Vector Machine)    — Powerful but slower, finds optimal boundaries

HOW THE ORIGINAL NOTEBOOK DID IT:
    - Cell 14: Trained all 3 models in a loop, compared accuracy
    - Result: Logistic Regression won with ~99.48% accuracy
    - Cell 12-13 (commented): Random Forest was tried earlier (98.4%)

USAGE:
    from src.model_training import train_and_select_best_model
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def get_models() -> dict:
    """
    Returns a dictionary of ML models to train and compare.

    You can add new models here in the future! Just add a new entry:
        "Random Forest": RandomForestClassifier(n_estimators=100),

    Returns:
        dict: {model_name: model_instance}
    """
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
    }

    # ---- OPTIONAL: Uncomment to include Random Forest ----
    # from sklearn.ensemble import RandomForestClassifier
    # models["Random Forest"] = RandomForestClassifier(n_estimators=100)


def train_and_select_best_model(x_train, x_test, y_train, y_test) -> tuple:
    """
    Train all models, evaluate them, and return the best one.

    Args:
        x_train: Training features (TF-IDF vectors)
        x_test:  Testing features
        y_train: Training labels (encoded categories)
        y_test:  Testing labels

    Returns:
        tuple: (best_model, best_model_name, best_accuracy, all_results)
            - best_model: The trained model object with highest accuracy
            - best_model_name: String name of the best model
            - best_accuracy: Float accuracy score of the best model
            - all_results: Dict of {name: {model, accuracy, report}} for all models

    Process:
        For each model:
        1. .fit(x_train, y_train)    → Learn patterns from training data
        2. .predict(x_test)          → Make predictions on unseen test data
        3. accuracy_score()          → Compare predictions vs actual answers
        4. classification_report()   → Detailed per-category precision/recall
    """
    models = get_models()
    all_results = {}
    best_model = None
    best_model_name = ""
    best_accuracy = 0

    print("=" * 60)
    print("🏋️  TRAINING MODELS")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n--- Training: {name} ---")

        # Step 1: Train the model
        model.fit(x_train, y_train)

        # Step 2: Make predictions on test data
        y_pred = model.predict(x_test)

        # Step 3: Calculate accuracy
        acc = accuracy_score(y_test, y_pred)

        # Step 4: Generate detailed report
        report = classification_report(y_test, y_pred)

        print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(report)

        # Store results
        all_results[name] = {
            "model": model,
            "accuracy": acc,
            "report": report,
            "predictions": y_pred,
        }

        # Track the best
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    print("=" * 60)
    print(f"🏆 BEST MODEL: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    print("=" * 60)

    return best_model, best_model_name, best_accuracy, all_results


# ============================================================
# Run directly to test: python -m src.model_training
# (requires data to be preprocessed first — use main.py instead)
# ============================================================
if __name__ == "__main__":
    print("This module is meant to be imported.")
    print("Run 'python main.py' for the full training pipeline.")
