# Data Handling
import pandas as pd
import numpy as np

# Paths
from src.__00__paths import curated_data_dir, processed_data_dir, raw_data_dir
import shutil
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

        # Check for an extra "Data" folder
        data_root = dataset_path / "Data" if (dataset_path / "Data").exists() else dataset_path

        # Copy files/folders to raw_data_dir
        for item in data_root.iterdir():
            target = raw_data_dir / item.name
            if item.is_file():
                shutil.copy2(item, target)
            elif item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)

        print("✔️ Dataset successfully downloaded.")


def data_preprocessing(df):
    """
    Encoding labels and scaling features.
    :param df: 3 seconds of audio features
    :return: preprocessed df, mapping_df
    """
    df.drop(['filename', 'length'], axis=1, inplace=True)

    # Encode Labels from 0 -> 9
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    # Save Label Mapping
    mapping_df = pd.DataFrame(label_encoder.classes_, columns=['Label'])

    # Separate features and label
    features = df.drop(columns=["label"])
    labels = df["label"]

    # Fit scaler on features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Reconstruct DataFrame
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled["label"] = labels
    # Replace original df with scaled version
    df = df_scaled

    return df, mapping_df


def split_dataset(df, test_size=0.2):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, test_df


def load_dataset(file_name):
    return pd.read_csv(file_name)


def save_dataset(df, file_name, index=False):
    pd.DataFrame(df).to_csv(file_name, index=index)
