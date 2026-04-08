from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .config import FEATURE_COLUMNS, RANDOM_SEED, SEQUENCE_LENGTH, TEST_SIZE


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    smile_thr = df["smile"].quantile(0.88)
    mouth_thr = df["mouth_open"].quantile(0.85)
    head_left_thr = -0.08
    head_right_thr = 0.08
    frontal_limit = 0.04

    labels = []
    for smile, mouth, head in zip(df["smile"], df["mouth_open"], df["head_turn"]):
        if head < head_left_thr:
            labels.append("head_left")
        elif head > head_right_thr:
            labels.append("head_right")
        elif abs(head) < frontal_limit and smile > smile_thr:
            labels.append("smiling")
        elif mouth > mouth_thr:
            labels.append("mouth_open")
        else:
            labels.append("neutral")

    df["label_name"] = labels
    label_to_id = {
        "neutral": 0,
        "smiling": 1,
        "mouth_open": 2,
        "head_left": 3,
        "head_right": 4,
    }
    df["label"] = df["label_name"].map(label_to_id)
    return df


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_seq, y_seq = [], []

    for i in range(sequence_length, len(features)):
        X_seq.append(features[i - sequence_length : i])
        y_seq.append(labels[i])

    return np.array(X_seq), np.array(y_seq)


def load_and_prepare_data(csv_path: str | bytes | object) -> PreparedData:
    df = pd.read_csv(csv_path)
    df = create_labels(df)

    features = df[FEATURE_COLUMNS].values
    labels = df["label"].values

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X, y = build_sequences(features_scaled, labels, SEQUENCE_LENGTH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
    )