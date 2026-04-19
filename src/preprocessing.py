"""
preprocessing.py — Text Cleaning & NLP Preprocessing
=====================================================

PURPOSE:
    This module contains all text cleaning logic for resume text. Raw resumes
    are messy — they contain URLs, special characters, numbers, common words
    ("the", "is", "and") that don't help classification. This module cleans
    all of that up so the ML model gets clean, meaningful text.

WHAT EACH STEP DOES (in order):
    1. Lowercase         → "Python" and "python" become the same word
    2. Remove URLs       → strips "http://..." links that don't help classification
    3. Remove non-alpha  → strips numbers, punctuation, special chars (keeps only letters)
    4. Tokenize          → splits text into individual words ["machine", "learning"]
    5. Remove stopwords  → removes common English words ("the", "is", "a", "an", "in")
    6. Lemmatize         → converts words to base form ("running" → "run", "better" → "good")

HOW THE ORIGINAL NOTEBOOK DID IT:
    - Cell 6 (imports): imported nltk, stopwords, WordNetLemmatizer
    - Cell 7 (clean_text function): the exact cleaning logic this module contains
    - Cell 8: Applied it to create the "Cleaned Resume" column

WHY STOPWORDS REMOVAL?
    Words like "the", "is", "and" appear in EVERY resume regardless of category.
    They don't help distinguish a "Data Science" resume from an "HR" resume.
    Removing them reduces noise and makes the model focus on meaningful words.

WHY LEMMATIZATION?
    "programming", "programmed", "programs" → all become "program"
    This reduces vocabulary size and groups related words together,
    helping the model recognize patterns better.

USAGE:
    from src.preprocessing import clean_text, preprocess_dataframe
    
    cleaned = clean_text("I am a Python developer with 5+ years experience")
    df = preprocess_dataframe(df)  # Adds "Cleaned Resume" column
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.config import RESUME_COLUMN, CLEANED_RESUME_COLUMN


def ensure_nltk_data():
    """
    Download required NLTK data packages if not already present.

    NLTK (Natural Language Toolkit) needs external data files for:
        - stopwords: list of common English words to remove
        - wordnet: dictionary database used for lemmatization

    This function is safe to call multiple times — it won't re-download
    if the data already exists.
    """
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)


# Initialize NLTK resources on module import
ensure_nltk_data()

# Create the stopwords set and lemmatizer ONCE (not inside every function call)
# set() is used instead of list because checking "if word in set" is O(1) vs O(n)
_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean a single resume text string.

    Args:
        text: Raw resume text (can contain URLs, special chars, mixed case, etc.)

    Returns:
        str: Cleaned text — lowercase, no URLs, no special chars, no stopwords,
             all words lemmatized.

    Example:
        >>> clean_text("I am a Python developer with http://linkedin.com/in/john")
        'python developer'
        
        (Note: "I", "am", "a", "with" are stopwords and get removed)

    Processing pipeline:
        "I am a Python Developer!!" 
        → "i am a python developer!!"           (lowercase)
        → "i am a python developer  "           (remove non-alpha)
        → ["i", "am", "a", "python", "developer"]  (split/tokenize)
        → ["python", "developer"]                (remove stopwords)
        → ["python", "developer"]                (lemmatize)
        → "python developer"                    (rejoin)
    """
    # Step 1: Lowercase everything
    text = text.lower()

    # Step 2: Remove URLs (http://..., https://..., www....)
    # \S+ matches any non-whitespace characters after "http"
    text = re.sub(r"http\S+", "", text)

    # Step 3: Remove all non-alphabetic characters (numbers, punctuation, symbols)
    # This replaces anything that's NOT a-z or A-Z with a space
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Step 4: Tokenize — split the string into a list of individual words
    words = text.split()

    # Step 5 & 6: Remove stopwords AND lemmatize in one pass
    # - "if word not in _stop_words" → keeps only meaningful words
    # - lemmatizer.lemmatize(word)   → converts to base form
    words = [
        _lemmatizer.lemmatize(word)
        for word in words
        if word not in _stop_words
    ]

    # Step 7: Join words back into a single string
    return " ".join(words)


def preprocess_dataframe(df):
    """
    Apply text cleaning to the entire DataFrame.

    Args:
        df: pandas DataFrame with a 'Resume' column containing raw text.

    Returns:
        pd.DataFrame: Same DataFrame with an added 'Cleaned Resume' column.

    What .apply() does:
        It runs clean_text() on EVERY row in the 'Resume' column, one by one.
        It's like a for-loop but optimized by pandas.
    """
    print("🧹 Cleaning resume text...")
    df[CLEANED_RESUME_COLUMN] = df[RESUME_COLUMN].apply(clean_text)
    print(f"✅ Created '{CLEANED_RESUME_COLUMN}' column")

    # Show a before/after sample
    print("\n--- Sample (before → after) ---")
    print(f"  Original : {df[RESUME_COLUMN].iloc[0][:80]}...")
    print(f"  Cleaned  : {df[CLEANED_RESUME_COLUMN].iloc[0][:80]}...")

    return df


# ============================================================
# Run directly to test: python -m src.preprocessing
# ============================================================
if __name__ == "__main__":
    test_texts = [
        "I am a Python developer with http://linkedin.com/in/john 5 years exp!",
        "Skills: Java, SQL, Machine Learning. GPA: 3.8/4.0",
        "Education Details \\r\\n MCA YMCAUST Faridabad",
    ]
    print("Testing clean_text():\n")
    for t in test_texts:
        print(f"  Input:  {t}")
        print(f"  Output: {clean_text(t)}")
        print()
