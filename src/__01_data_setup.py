# Data Handling
import pandas as pd
import numpy as np

# Paths
from src.__00__paths import curated_data_dir, processed_data_dir, raw_data_dir
import shutil
from pathlib import Path

# Datasets Source
import kagglehub


def download_dataset():
    # List of files to check
    data_items = [
        raw_data_dir / "features_3_sec.csv",
        raw_data_dir / "features_30_sec.csv",
        raw_data_dir / "genres_original",
        raw_data_dir / "images_original",
    ]

    # Check and download
    if all(item.exists() for item in data_items):
        print("✔️ Dataset is already downloaded.")
    else:

        # Download dataset
        dataset_path = Path(kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification"))

        if not dataset_path.exists():
            raise FileNotFoundError("⚠️ Dataset not found.")

        # Copy files to raw_data_dir
        for item in dataset_path.iterdir():
            target = raw_data_dir / item.name
            if item.is_file():
                shutil.copy2(item, target)
            elif item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)

        print("✔️ Dataset successfully downloaded.")


def load_dataset(file_name):
    return pd.read_csv(file_name)
