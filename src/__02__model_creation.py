# Standard Library
from pathlib import Path

# Third-Party Libraries
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
from xgboost import XGBClassifier
import joblib

# Local Modules
from src.__00__paths import model_dir


class ChannelAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class Genre_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def block(in_c, out_c, drop):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                ChannelAttention(out_c),
                nn.MaxPool2d(2),
                nn.Dropout(drop)
            )

        self.encoder = nn.Sequential(
            block(1, 32, 0.25),
            block(32, 64, 0.25),
            block(64, 128, 0.3),
            block(128, 256, 0.3),
            block(256, 512, 0.4)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)


class GenreSpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {genre.name: idx for idx, genre in enumerate(sorted(self.root_dir.iterdir()))}

        for genre in self.class_to_idx:
            for file in (self.root_dir / genre).glob("*.png"):
                self.samples.append((file, self.class_to_idx[genre]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label


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
