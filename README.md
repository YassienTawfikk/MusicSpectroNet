# MusicSpectroNet

> GTZAN Genre Classification Using Tabular ML and CNN Architectures

<p align='center'>
   <img width="500" alt="20250808_2111_MusicSpectroNet Poster_simple_compose_01k25e2jdweta9vj2y020y8z46" src="https://github.com/user-attachments/assets/9c70cda3-dd4a-4f54-9557-af484d182792" />
</p>

---

## Problem Statement

Music genre classification plays a key role in music recommendation, audio indexing, and musicological research. Despite recent advances in deep learning and audio processing, identifying optimal architectures and input representations remains challenging.

This project investigates and compares the performance of two modeling approaches on the GTZAN dataset:

* **Tabular ML** using extracted audio features from 3-second segments.
* **CNN** using spectrogram images.

---

## Dataset Overview

We used the [GTZAN Music Genre Classification dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), a well-known benchmark dataset that includes:

* `genres_original/`: 10 music genres with 100 audio clips each (30s `.wav` files)
* `images_original/`: Mel Spectrogram image representation of the audio files
* `features_30_sec.csv`: Mean/variance of audio features extracted from full 30-second clips
* `features_3_sec.csv`: Same features but split into \~10 x 3-second chunks per clip (\~9990 samples)

---

## Features

We relied on the `features_3_sec.csv`, containing 58 numerical features:

| Feature Group       | Description                                                       |
| ------------------- | ----------------------------------------------------------------- |
| Chroma Features     | `chroma_stft_mean`, `chroma_stft_var`                             |
| RMS Energy          | `rms_mean`, `rms_var`                                             |
| Spectral Features   | `spectral_centroid`, `bandwidth`, `rolloff`, `zero_crossing_rate` |
| Harmonic/Perceptual | `harmony`, `perceptr_mean`, `perceptr_var`                        |
| Tempo               | `tempo`                                                           |
| MFCCs               | `mfcc1_mean` to `mfcc20_var`                                      |
| Label               | Genre name or class index                                         |

The dataset was cleaned, encoded, and normalized for input to ML models.

---

## Model Comparison: Tabular ML vs. CNN

We explored both approaches:

### Final Comparison Table

| Metric        | Tabular ML (XGBoost) | CNN (Mel Spectrograms) |
| ------------- | -------------------- | ---------------------- |
| Test Accuracy | **92.64%**           | 23.33%                 |
| Input Format  | Extracted features   | Image                  |
| Train Samples | 7992 (tabular)       | 1200 (images)          |
| Model Type    | XGBoost (tree-based) | Custom CNN             |

> Confusion matrix visualizations are saved under `outputs/figures/`

### Performance Insights

We saved and visualized both confusion matrices and feature importances:

| Confusion Matrix (XGBoost) | Confusion Matrix (CNN) |
| ---------------------- | -------------------------- |
| <img width="640" height="480" alt="xgboost_confusion_matrix" src="https://github.com/user-attachments/assets/d7038408-1297-4db0-86dc-1ee810b952f7" /> | <img width="640" height="480" alt="cnn_confusion_matrix" src="https://github.com/user-attachments/assets/2db6c7f4-56aa-4ea4-96c3-2cf9e9fee4a7" /> |

---

### Feature Importance (XGBoost)

Top features include:

* perceptr\_var – Perceptual spread of sound
* spectral\_bandwidth\_mean – Frequency range
* chroma\_stft\_mean – Tonal content
* Key MFCCs (mfcc1, mfcc4, mfcc9) – Shape of sound spectrum

<p align='center'>
<img width="640" height="480" alt="xgboost_feature_importance" src="https://github.com/user-attachments/assets/44b5ebd1-9707-45b5-905a-708921acd135" />
</p>

> XGBoost focused on features that capture texture, brightness, and energy of music — all crucial for genre detection.

---

## Conclusion

* The CNN model suffered from improper architecture and lacked optimization, achieving only 23.33% test accuracy.
* XGBoost using tabular data (features extracted from Librosa) achieved a strong 92.64% accuracy.
* **Our primary focus is on tabular ML**, as it demonstrated better performance and robustness on this task.

---

## Project Structure

```
MusicSpectroNet/
│
├── documents/
│   └── project_structure.txt
│
├── data/
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Processed features/labels
│   └── curated/               # Final train/test splits
│
├── spectronet/                # Main package
│   ├── __init__.py
│   ├── data.py                # Data loading and preprocessing
│   ├── models.py              # Model definitions (CNN & XGBoost)
│   └── paths.py               # Path configurations
│
├── scripts/
│   └── training.py            # CLI entry point for training
│
├── run/
│   └── train.sh               # SLURM-compatible launch script
│
├── outputs/
│   ├── models/
│   ├── figures/
│   └── docs/
│
├── README.md                  # Project documentation
├── requirements.txt           # Dependencies
└── .gitignore
```

---

## How to Run

1. Clone the repo and install requirements:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the training pipeline via the convenience script:

   ```bash
   bash run/train.sh
   ```

   This script handles the environment setup and launches the training CLI.

3. Alternatively, run the CLI directly:

   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   python scripts/training.py fit \
      --n_estimators=1000 \
      --learning_rate=0.05 \
      --max_depth=6
   ```

---

## Submission

This project was developed as part of the **MusicSpectroNet Series**, showcasing real-world experimentation between CNN-based image learning and tabular audio classification — with insights drawn from both performance and engineering complexity.

---

## Author

<div>
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/YassienTawfikk" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126521373?v=4" width="150px;" alt="Yassien Tawfik"/>
        <br>
        <sub><b>Yassien Tawfik</b></sub>
      </a>
    </td>
  </tr>
</table>
</div>
