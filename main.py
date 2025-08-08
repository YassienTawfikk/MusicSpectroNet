from src.__00__paths import raw_data_dir, curated_tabular_dir, processed_tabular_dir, model_dir
from src.__01_data_setup import load_dataset, data_preprocessing, save_dataset, split_dataset, download_dataset
from src.__02__model_creation import save_model, setup_model, train_model


def main():
    print(f"Creating Needed Directories...")

    print(f"Download Datasets...")
    download_dataset()

    print(f"Loading Datasets...")
    raw_3_sec_data = load_dataset(raw_data_dir / "features_3_sec.csv")

    print(f"Preprocessing Datasets...")
    processed_df, labels_df = data_preprocessing(raw_3_sec_data)

    print("Splitting Dataset...")
    train_df, test_df = split_dataset(processed_df)

    print(f"Saving Datasets...")
    save_dataset(processed_df, processed_tabular_dir / "features_3_sec_processed.csv", index=False)
    save_dataset(labels_df, processed_tabular_dir / "labels_3_sec_processed.csv", index=True)
    save_dataset(train_df, curated_tabular_dir / "train.csv", index=False)
    save_dataset(test_df, curated_tabular_dir / "test.csv", index=False)

    print("✔️ Data setup complete. (Preprocessing - Splitting - Saving)")

    print("Setting up Model...")
    model = setup_model()

    print("Train Model...")
    train_model(model, train_df.drop['label'], train_df['label'])
    print(f"✔️ Model Created Sucessfully)")

    print("Saving Model...")
    save_model(model, model_dir / "xgboost_gtzan_model.pkl")
    print("✔️ Model Saved.")


if __name__ == "__main__":
    main()
