import joblib
from src.__00__paths import model_dir
from xgboost import XGBClassifier


def save_model(model, file_name=model_dir / "genre_cnn_model.pth"):
    joblib.dump(model, file_name)
    print(f"✔️ Model Saved at {'/'.join(file_name.parts[-2:])}")


def setup_model(n_estimators=1000, learning_rate=0.05, max_depth=6, num_class=10):
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=num_class,
        verbosity=1,
        random_state=42,
        eval_metric="mlogloss"
    )
    return model


def train_model(model, features, labels):
    model.fit(features, labels)
