"""
skills_extractor.py — Keyword-Based Skill Extraction
=====================================================

PURPOSE:
    Extracts technical skills from resume text by matching against a
    predefined database of skill keywords. This is a simple but effective
    approach — it checks if each known skill appears in the resume text.

HOW THE ORIGINAL NOTEBOOK DID IT:
    - Cell 16: Defined skills_db list (10 skills) and extract_skills() function
    - We expand the skills database to 40+ skills and make it configurable

FUTURE IMPROVEMENTS:
    - Use NLP-based entity recognition (spaCy NER) for smarter extraction
    - Load skills from an external file (JSON/CSV) for easy updates
    - Add skill categorization (programming, frameworks, tools, etc.)

USAGE:
    from src.skills_extractor import extract_skills
    
    skills = extract_skills("I know Python and machine learning")
    # Returns: ['python', 'machine learning']
"""

from src.config import SKILLS_DB


def extract_skills(text: str, skills_list: list = None) -> list:
    """
    Extract skills from resume text using keyword matching.

    Args:
        text: Raw or cleaned resume text.
        skills_list: Optional custom list of skills to search for.
                     Defaults to SKILLS_DB from config.py.

    Returns:
        list: Skills found in the text (lowercase).

    How it works:
        1. Convert text to lowercase (so "Python" matches "python")
        2. For each skill in the database, check if it appears in the text
        3. Collect and return all matches

    Example:
        >>> extract_skills("Expert in Python, SQL, and machine learning")
        ['python', 'machine learning', 'sql']
    """
    if skills_list is None:
        skills_list = SKILLS_DB

    text_lower = text.lower()
    found_skills = []

    for skill in skills_list:
        if skill in text_lower:
            found_skills.append(skill)

    return found_skills


# ============================================================
# Run directly to test: python -m src.skills_extractor
# ============================================================
if __name__ == "__main__":
    test_resumes = [
        "Expert Python developer with machine learning and deep learning experience. Proficient in SQL and pandas.",
        "Java developer with Spring Boot, Docker, and Jenkins CI/CD experience.",
        "Data analyst skilled in Excel, Tableau, Power BI, and SQL databases.",
    ]

    print("Testing skill extraction:\n")
    for resume in test_resumes:
        skills = extract_skills(resume)
        print(f"  Resume: {resume[:60]}...")
        print(f"  Skills: {skills}\n")
