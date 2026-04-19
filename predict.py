"""
predict.py — Prediction Entry Point
=====================================

PURPOSE:
    Standalone script to predict the job category of a resume.
    Uses pre-trained models (from running main.py) to make predictions
    WITHOUT retraining.

HOW TO RUN:
    python predict.py

    Make sure you've run 'python main.py' first to train and save models!

WHAT THIS REPLACES:
    The sample resume prediction in the last cells of Untitled15.ipynb.
"""

from src.predictor import ResumePredictor


def main():
    """Run interactive resume prediction."""
    print("=" * 60)
    print("🔍 RESUME ANALYZER — PREDICTION MODE")
    print("=" * 60)

    # Load pre-trained models
    print("\nLoading trained models...")
    predictor = ResumePredictor()

    # ---- Sample Resume (from notebook) ----
    sample_resume = """
    I am a computer science student skilled in Python, machine learning, and data analysis.
    Worked on deep learning projects and SQL databases.
    """

    print("\n" + "-" * 40)
    print("📄 Sample Resume:")
    print(sample_resume.strip())
    print("-" * 40)

    result = predictor.predict(sample_resume)

    print(f"\n🎯 Predicted Role: {result['predicted_role']}")
    print(f"🛠️  Extracted Skills: {result['skills']}")

    # ---- Interactive Mode ----
    print("\n" + "=" * 60)
    print("📝 INTERACTIVE MODE")
    print("   Paste a resume below and press Enter twice to predict.")
    print("   Type 'quit' to exit.")
    print("=" * 60)

    while True:
        print("\n📄 Enter resume text (or 'quit'):")
        lines = []
        while True:
            line = input()
            if line.strip().lower() == "quit":
                print("\n👋 Goodbye!")
                return
            if line == "" and lines:
                break
            lines.append(line)

        resume_text = "\n".join(lines)

        if not resume_text.strip():
            print("⚠️  Empty input. Please enter some resume text.")
            continue

        result = predictor.predict(resume_text)

        print(f"\n🎯 Predicted Role: {result['predicted_role']}")
        print(f"🛠️  Extracted Skills: {result['skills']}")


if __name__ == "__main__":
    main()
