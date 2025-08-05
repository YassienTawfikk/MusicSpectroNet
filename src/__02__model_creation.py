from sklearn.ensemble import RandomForestRegressor
import joblib
from src.__00__paths import model_dir


def save_model(model, file_name=model_dir / "random_forest_model.joblib"):
    joblib.dump(model, file_name)
    print(f"✔️ Model Saved at {'/'.join(file_name.parts[-2:])}")
