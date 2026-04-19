"""
Resume Analyzer — Source Package
================================

This package contains all the core modules for the Resume Analyzer project:

Modules:
    config              - Central configuration (paths, constants, hyperparameters)
    data_loader         - Download and load the Kaggle resume dataset
    preprocessing       - Text cleaning (stopwords removal, lemmatization, regex)
    feature_engineering - Label encoding + TF-IDF vectorization + train/test split
    model_training      - Train multiple ML models, compare, and select the best
    model_persistence   - Save/load trained models and vectorizers using pickle
    skills_extractor    - Keyword-based skill extraction from resume text
    predictor           - End-to-end prediction pipeline (clean → vectorize → predict)
    visualization       - Optional plotting utilities for EDA and evaluation
"""
