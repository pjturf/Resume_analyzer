"""
config.py — Central Configuration
==================================

PURPOSE:
    This file stores ALL configuration values for the entire project in ONE place.
    Instead of scattering magic numbers, file paths, and settings across multiple files,
    we centralize them here. When you need to change a setting (e.g., increase TF-IDF
    features from 5000 to 10000), you change it HERE and it applies everywhere.

WHY THIS MATTERS:
    - Avoids hardcoded paths like "C:\\Users\\Prachi\\.cache\\..." scattered in code
    - Makes the project portable (works on any machine, not just yours)
    - Single source of truth for all hyperparameters
    - Easy to tweak settings when experimenting

USAGE:
    from src.config import DATASET_NAME, DATA_DIR, TFIDF_MAX_FEATURES
"""

import os

# ============================================================
# PATHS
# ============================================================

# Root directory of the project (parent of this file's folder)
# os.path.dirname(__file__)         → gets the "src/" folder path
# os.path.dirname(os.path.dirname)  → goes one level up to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where downloaded datasets are stored
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Where trained model files (.pkl) are saved
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ============================================================
# DATASET
# ============================================================

# Kaggle dataset identifier (used by kagglehub to download)
# This is the "jillanisofttech/updated-resume-dataset" from Kaggle
KAGGLE_DATASET = "jillanisofttech/updated-resume-dataset"

# Name of the CSV file inside the downloaded dataset
DATASET_FILENAME = "UpdatedResumeDataSet.csv"

# ============================================================
# COLUMN NAMES (from the CSV)
# ============================================================

# The column containing the job category label (e.g., "Data Science", "HR")
CATEGORY_COLUMN = "Category"

# The column containing the raw resume text
RESUME_COLUMN = "Resume"

# The column we create after cleaning the resume text
CLEANED_RESUME_COLUMN = "Cleaned Resume"

# The column we create after encoding categories to numbers
ENCODED_CATEGORY_COLUMN = "Category_Encoded"

# ============================================================
# FEATURE ENGINEERING HYPERPARAMETERS
# ============================================================

# Maximum number of TF-IDF features to extract
# Higher = more features = potentially better accuracy but slower training
# Your notebook used 5000, which is a solid default
TFIDF_MAX_FEATURES = 5000

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

# Fraction of data reserved for testing (0.2 = 20%)
TEST_SIZE = 0.2

# Random seed for reproducibility — ensures the same split every time
# Your notebook used 42, which is the universal convention
RANDOM_STATE = 42

# ============================================================
# MODEL FILE NAMES (saved in MODELS_DIR)
# ============================================================

# Filename for the saved TF-IDF vectorizer
TFIDF_MODEL_FILE = "tfidf_vectorizer.pkl"

# Filename for the saved label encoder
LABEL_ENCODER_FILE = "label_encoder.pkl"

# Filename for the saved best classifier model
BEST_MODEL_FILE = "best_model.pkl"

# ============================================================
# SKILLS DATABASE
# ============================================================

# List of skills to search for in resume text.
# This is a simple keyword-matching approach.
# You can expand this list as the project grows.
SKILLS_DB = [
    "python", "java", "machine learning", "sql", "html",
    "css", "javascript", "data analysis", "deep learning", "excel",
    "react", "node.js", "docker", "kubernetes", "aws",
    "tensorflow", "pytorch", "pandas", "numpy", "git",
    "mongodb", "postgresql", "mysql", "flask", "django",
    "spring boot", "c++", "c#", "ruby", "go",
    "tableau", "power bi", "hadoop", "spark", "scala",
    "natural language processing", "computer vision", "api",
    "linux", "agile", "scrum", "jira", "jenkins",
]
