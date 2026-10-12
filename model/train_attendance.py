"""
Train a TensorFlow/Keras classifier for employee attendance risk.

Run from the project root: `python model/train_attendance.py`

Saves:
  model/artifacts/attendance_preprocessor.joblib   sklearn ColumnTransformer
  model/artifacts/attendance_model.keras            trained Keras model
  model/artifacts/attendance_metrics.json           metrics + training curves
  model/artifacts/attendance_background.npy         background sample for SHAP
"""

import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

tf.random.set_seed(42)

NUMERIC = ["attendance_rate", "unplanned_absences", "sick_days", "approved_leave_days",
           "wfh_days", "late_arrivals", "overtime_hours", "avg_checkin_delay_min", "tenure_months"]
CATEGORICAL = ["seniority", "department", "role"]
FEATURE_COLS = NUMERIC + CATEGORICAL
TARGET = "high_absenteeism_risk"


def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model


def main():
    df = pd.read_csv("data/attendance_monthly.csv")
    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    X_train_t = prep.fit_transform(X_train).astype("float32")
    X_val_t = prep.transform(X_val).astype("float32")
    X_test_t = prep.transform(X_test).astype("float32")

    class_weight = {0: 1.0, 1: (y_train == 0).sum() / max(1, (y_train == 1).sum())}
    print(f"class_weight: {class_weight}")

    model = build_model(X_train_t.shape[1])
    history = model.fit(
        X_train_t, y_train.values, validation_data=(X_val_t, y_val.values),
        epochs=60, batch_size=32, class_weight=class_weight, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_auc", mode="max")],
    )

    # Class weighting during training skews raw outputs away from true probabilities
    # (they stop reflecting the real ~14% base rate), so calibrate them back with
    # isotonic regression fit on the validation set before using them for anything
    # downstream — thresholds, risk tiers, or just the number shown to a user.
    val_proba_raw = model.predict(X_val_t, verbose=0).flatten()
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(val_proba_raw, y_val.values)

    # Pick the operating threshold by maximizing F1 on the (calibrated) validation
    # set — the standard way to choose a decision point for an imbalanced classifier,
    # rather than defaulting to 0.5, which is meaningless once probabilities are
    # recalibrated to a ~14% base rate.
    val_proba_cal = calibrator.transform(val_proba_raw)
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_val, (val_proba_cal >= t).astype(int)) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1s))])

    proba_raw = model.predict(X_test_t, verbose=0).flatten()
    proba = calibrator.transform(proba_raw)
    pred = (proba >= best_threshold).astype(int)
    tier_cutoffs = {"low_max": float(np.quantile(val_proba_cal, 0.70)),
                     "medium_max": float(np.quantile(val_proba_cal, 0.90))}
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "decision_threshold": best_threshold,
        "tier_cutoffs": tier_cutoffs,
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "positive_rate": float(y.mean()),
        "epochs_trained": len(history.history["loss"]),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    fpr, tpr, _ = roc_curve(y_test, proba)
    idx = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)
    metrics["roc_curve"] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}
    metrics["confusion_matrix"] = {"labels": ["low_risk", "high_risk"], "matrix": confusion_matrix(y_test, pred).tolist()}
    print(metrics["roc_auc"], metrics["accuracy"], metrics["precision"], metrics["recall"])

    joblib.dump(prep, "model/artifacts/attendance_preprocessor.joblib")
    joblib.dump(calibrator, "model/artifacts/attendance_calibrator.joblib")
    model.save("model/artifacts/attendance_model.keras")
    with open("model/artifacts/attendance_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Background sample for SHAP (model-agnostic Explainer needs a reference distribution)
    bg_idx = np.random.default_rng(42).choice(len(X_train_t), size=min(100, len(X_train_t)), replace=False)
    np.save("model/artifacts/attendance_background.npy", X_train_t[bg_idx])

    # Holdout sample for the app's batch-scoring demo
    sample = X_test.copy()
    sample[TARGET] = y_test.values
    sample.sample(n=min(150, len(sample)), random_state=42).to_csv(
        "model/artifacts/attendance_sample.csv", index=False
    )
    print("Saved attendance model artifacts.")


if __name__ == "__main__":
    main()
