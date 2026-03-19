from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "male", "age", "education", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

TARGET = "TenYearCHD"


@dataclass(frozen=True)
class TrainedModel:
    model: LogisticRegression
    scaler: StandardScaler
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    confusion: np.ndarray
    report: str
    y_test: np.ndarray
    y_proba: np.ndarray


def load_framingham_csv(path_or_buf):
    df = pd.read_csv(path_or_buf)
    missing = [c for c in (FEATURES + [TARGET]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def _fill_missing_with_mean(df):
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].fillna(df[numeric].mean())
    return df


def train_from_dataframe(df, random_state=2):
    df = _fill_missing_with_mean(df)

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # ✅ 70% TRAIN, 30% TEMP
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )

    # ✅ 15% VALIDATION, 15% TEST
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Probabilities (for ROC)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    conf = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, digits=3)

    return TrainedModel(
        model=model,
        scaler=scaler,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        test_accuracy=test_acc,
        confusion=conf,
        report=report,
        y_test=y_test.values,
        y_proba=y_proba
    )


def predict_one(trained: TrainedModel, features: dict):
    x = pd.DataFrame([[features[f] for f in FEATURES]], columns=FEATURES)
    x_scaled = trained.scaler.transform(x)

    pred = int(trained.model.predict(x_scaled)[0])
    proba = float(trained.model.predict_proba(x_scaled)[0, 1])

    return pred, proba
