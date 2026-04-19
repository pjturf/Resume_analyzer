"""
data_loader.py — Dataset Acquisition
======================================

PURPOSE:
    This module handles downloading the resume dataset from Kaggle and loading
    it into a pandas DataFrame. It replaces the hardcoded path from the notebook
    (C:\\Users\\Prachi\\.cache\\...) with a dynamic, portable approach.

WHAT IT DOES:
    1. Uses `kagglehub` to download the dataset from Kaggle
    2. Finds the CSV file in the downloaded directory
    3. Loads it into a pandas DataFrame
    4. Optionally copies the CSV to the project's data/ folder for easy access

HOW THE ORIGINAL NOTEBOOK DID IT:
    - Cell 1: !pip install kagglehub → then kagglehub.dataset_download(...)
    - Cell 2: pd.read_csv(r"C:\\Users\\Prachi\\.cache\\kagglehub\\...")
    
    The problem: the path was hardcoded to YOUR machine. This module fixes that.

USAGE:
    from src.data_loader import download_dataset, load_dataset
    
    path = download_dataset()          # Downloads from Kaggle
    df = load_dataset()                # Downloads + loads into DataFrame
"""

import os
import shutil
import pandas as pd
import kagglehub

from src.config import (
    KAGGLE_DATASET,
    DATASET_FILENAME,
    DATA_DIR,
    CATEGORY_COLUMN,
    RESUME_COLUMN,
)


def download_dataset() -> str:
    """
    Download the resume dataset from Kaggle using kagglehub.

    Returns:
        str: Path to the directory containing the downloaded dataset files.

    How it works:
        kagglehub.dataset_download() checks if the dataset is already cached.
        If yes, it returns the cached path. If no, it downloads it first.
        This means running it twice won't re-download — it's smart about caching.
    """
    print("📥 Downloading dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"✅ Dataset available at: {dataset_path}")
    return dataset_path


def load_dataset(copy_to_data_dir: bool = True) -> pd.DataFrame:
    """
    Download the dataset and load it into a pandas DataFrame.

    Args:
        copy_to_data_dir: If True, copies the CSV into the project's data/ folder
                          so you have a local copy independent of the kagglehub cache.

    Returns:
        pd.DataFrame: The loaded resume dataset with columns ['Category', 'Resume'].

    Steps:
        1. Download (or get cached path) via kagglehub
        2. Build the full path to the CSV file
        3. Read it into a DataFrame
        4. Optionally copy to data/ for convenience
        5. Print basic info (shape, columns)
    """
    # Step 1: Download / get cached path
    dataset_dir = download_dataset()

    # Step 2: Build full CSV path
    csv_path = os.path.join(dataset_dir, DATASET_FILENAME)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"❌ Expected CSV file not found at: {csv_path}\n"
            f"Files in directory: {os.listdir(dataset_dir)}"
        )

    # Step 3: Load into DataFrame
    print(f"📖 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Step 4: Optionally copy to project's data/ folder
    if copy_to_data_dir:
        os.makedirs(DATA_DIR, exist_ok=True)
        local_csv_path = os.path.join(DATA_DIR, DATASET_FILENAME)
        if not os.path.exists(local_csv_path):
            shutil.copy2(csv_path, local_csv_path)
            print(f"📋 Copied dataset to: {local_csv_path}")

    # Step 5: Print basic info
    print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Categories: {df[CATEGORY_COLUMN].nunique()} unique")

    return df


# ============================================================
# Run directly to test: python -m src.data_loader
# ============================================================
if __name__ == "__main__":
    df = load_dataset()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nCategory distribution:")
    print(df[CATEGORY_COLUMN].value_counts())
