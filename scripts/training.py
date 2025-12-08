import sys
from jsonargparse import CLI
from spectronet.paths import raw_data_dir, processed_tabular_dir, curated_tabular_dir, model_dir
from spectronet.data import download_dataset, load_dataset, data_preprocessing, split_dataset, save_dataset
from spectronet.models import setup_xgb, train_model, save_model

class TrainingCLI:
    def fit(
        self,
        data_path: str = str(raw_data_dir / "features_3_sec.csv"),
        test_size: float = 0.2,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_class: int = 10,
    ):
        """
        Train the Music SpectroNet model.
        
        Args:
            data_path: Path to the raw features CSV file.
            test_size: Fraction of data to use for testing.
            n_estimators: Number of XGBoost estimators.
            learning_rate: XGBoost learning rate.
            max_depth: XGBoost max depth.
            num_class: Number of classes.
        """
        print(f"Creating Needed Directories...")

        print(f"Download Datasets...")
        download_dataset()

        print(f"Loading Datasets...")
        raw_3_sec_data = load_dataset(data_path)

        print(f"Preprocessing Datasets...")
        processed_df, labels_df = data_preprocessing(raw_3_sec_data)

        print("Splitting Dataset...")
        train_df, test_df = split_dataset(processed_df, test_size=test_size)

        print(f"Saving Datasets...")
        save_dataset(processed_df, processed_tabular_dir / "features_3_sec_processed.csv", index=False)
        save_dataset(labels_df, processed_tabular_dir / "labels_3_sec_processed.csv", index=True)
        save_dataset(train_df, curated_tabular_dir / "train.csv", index=False)
        save_dataset(test_df, curated_tabular_dir / "test.csv", index=False)

        print("✔️ Data setup complete. (Preprocessing - Splitting - Saving)")

        print("Setting up Model...")
        model = setup_xgb(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_class=num_class
        )

        print("Train Model...")
        train_model(model, train_df.drop(['label'], axis=1), train_df['label'])
        print(f"✔️ Model Created Successfully")

        print("Saving Model...")
        save_model(model, model_dir / "xgboost_gtzan_model.pkl")
        print("✔️ Model Saved.")

if __name__ == "__main__":
    CLI(TrainingCLI)
