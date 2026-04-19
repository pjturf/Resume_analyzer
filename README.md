# 📄 Resume Analyzer

An ML-powered resume classification system that predicts the job category of a resume and extracts relevant skills.

## 🎯 What It Does

Given raw resume text, this project:
1. **Cleans** the text (removes URLs, special chars, stopwords; lemmatizes words)
2. **Classifies** the resume into one of **25 job categories** (Data Science, Java Developer, HR, Testing, etc.)
3. **Extracts skills** mentioned in the resume (Python, SQL, Machine Learning, etc.)

## 📊 Model Performance

| Model               | Accuracy |
|---------------------|----------|
| Naive Bayes          | 98.96%   |
| Logistic Regression  | **99.48%** ✅ |
| SVM                  | 99.48%   |

The best model (Logistic Regression) is automatically selected and saved.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python main.py
```
This downloads the dataset from Kaggle, trains 3 models, and saves the best one.

### 3. Predict
```bash
python predict.py
```
This loads the trained model and lets you classify resumes interactively.

## 📁 Project Structure

```
Resume_analyzer/
├── data/                       # Downloaded datasets (auto-populated)
├── models/                     # Saved trained models (.pkl files)
├── src/                        # Core source code package
│   ├── __init__.py             # Package initializer
│   ├── config.py               # All settings, paths, constants
│   ├── data_loader.py          # Download & load Kaggle dataset
│   ├── preprocessing.py        # Text cleaning (stopwords, lemmatization)
│   ├── feature_engineering.py  # Label encoding + TF-IDF vectorization
│   ├── model_training.py       # Train & compare ML models
│   ├── model_persistence.py    # Save/load models with pickle
│   ├── skills_extractor.py     # Keyword-based skill extraction
│   ├── predictor.py            # End-to-end prediction pipeline
│   └── visualization.py        # Plotting utilities (optional)
├── main.py                     # Training entry point
├── predict.py                  # Prediction entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🗂️ Job Categories (25)

Advocate, Arts, Automation Testing, Blockchain, Business Analyst, Civil Engineer, Data Science, Database, DevOps Engineer, DotNet Developer, ETL Developer, Electrical Engineering, HR, Hadoop, Health and fitness, Java Developer, Mechanical Engineer, Network Security Engineer, Operations Manager, PMO, Python Developer, SAP Developer, Sales, Testing, Web Designing

## 📦 Dataset

- **Source**: [Kaggle — Updated Resume Dataset](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset)
- **Size**: 962 resumes across 25 categories
- **Columns**: `Category` (job role), `Resume` (raw text)

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas** — Data manipulation
- **NLTK** — Text preprocessing (stopwords, lemmatization)
- **scikit-learn** — ML models, TF-IDF, evaluation metrics
- **kagglehub** — Dataset download
- **matplotlib & seaborn** — Visualization (optional)
