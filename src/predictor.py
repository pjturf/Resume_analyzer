"""
predictor.py — End-to-End Resume Prediction Pipeline
=====================================================

PURPOSE:
    This is the inference module. Given raw resume text, it:
    1. Loads the pre-trained model, TF-IDF vectorizer, and label encoder
    2. Cleans the resume text
    3. Converts it to TF-IDF features
    4. Predicts the job category
    5. Extracts skills
    6. Returns results

HOW THE ORIGINAL NOTEBOOK DID IT:
    - Cell 17-18: predict_resume() function that combined cleaning,
      vectorization, prediction, and skill extraction

USAGE:
    from src.predictor import ResumePredictor
    
    predictor = ResumePredictor()
    result = predictor.predict("I am a Python developer...")
    print(result["predicted_role"])   # "Python Developer"
    print(result["skills"])           # ["python", ...]
"""

from src.preprocessing import clean_text
from src.skills_extractor import extract_skills
from src.model_persistence import load_artifacts


class ResumePredictor:
    """
    End-to-end resume prediction pipeline.

    Loads pre-trained artifacts once and reuses them for multiple predictions.
    This is more efficient than loading from disk every time.

    Attributes:
        model:            Trained classifier (e.g., LogisticRegression)
        tfidf_vectorizer: Fitted TfidfVectorizer for text → numbers
        label_encoder:    Fitted LabelEncoder for number → category name
    """

    def __init__(self):
        """Load pre-trained model artifacts from disk."""
        self.model, self.tfidf_vectorizer, self.label_encoder = load_artifacts()

    def predict(self, resume_text: str) -> dict:
        """
        Predict the job category and extract skills from resume text.

        Args:
            resume_text: Raw resume text (can be messy, uncleaned).

        Returns:
            dict with keys:
                - predicted_role: The predicted job category (string)
                - skills: List of extracted skills
                - cleaned_text: The cleaned version of the resume
                - confidence_note: Note about the prediction

        Pipeline:
            raw text → clean_text() → tfidf.transform() → model.predict()
                                                        → le.inverse_transform()
        """
        # Step 1: Clean the resume text
        cleaned = clean_text(resume_text)

        # Step 2: Convert to TF-IDF features
        # Note: we use .transform() NOT .fit_transform()
        # fit_transform() would re-learn vocabulary from scratch
        # transform() uses the ALREADY LEARNED vocabulary from training
        vector = self.tfidf_vectorizer.transform([cleaned])

        # Step 3: Predict the category (returns encoded number)
        prediction = self.model.predict(vector)

        # Step 4: Convert number back to category name
        role = self.label_encoder.inverse_transform(prediction)[0]

        # Step 5: Extract skills from the ORIGINAL text (not cleaned)
        skills = extract_skills(resume_text)

        return {
            "predicted_role": role,
            "skills": skills,
            "cleaned_text": cleaned,
        }


def predict_resume(resume_text: str) -> tuple:
    """
    Convenience function for quick predictions (matches notebook API).

    Args:
        resume_text: Raw resume text.

    Returns:
        tuple: (role, skills) — same as the notebook's predict_resume()
    """
    predictor = ResumePredictor()
    result = predictor.predict(resume_text)
    return result["predicted_role"], result["skills"]
