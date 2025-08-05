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
    data_items = [
        raw_data_dir / "features.csv",
        raw_data_dir / "stores.csv",
        raw_data_dir / "test.csv",
        raw_data_dir / "train.csv"
    ]

    if all(item.exists() for item in data_items):
        print("✔️ Data already downloaded")
    else:
        dataset_path = Path(kagglehub.dataset_download("aslanahmedov/walmart-sales-forecast"))

        if not dataset_path.exists():
            raise FileNotFoundError("Dataset not found")

        for item in dataset_path.iterdir():
            target = raw_data_dir / item.name
            shutil.copy2(item, target)

        print("✔️ Data downloaded successfully")


def load_dataset(file_name):
    return pd.read_csv(file_name)
