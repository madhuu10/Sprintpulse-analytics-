"""
Train a TensorFlow/Keras regressor predicting sprint completion rate.

Run from the project root: `python model/train_sprint.py`

Saves:
  model/artifacts/sprint_preprocessor.joblib
  model/artifacts/sprint_model.keras
  model/artifacts/sprint_metrics.json
  model/artifacts/sprint_background.npy
  model/artifacts/sprint_sample.csv
"""

import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

tf.random.set_seed(42)

NUMERIC = ["team_size", "avg_team_attendance_rate", "standup_attendance_pct", "carryover_points_in",
           "scope_change_pct", "test_coverage_pct", "code_review_pass_rate", "bugs_found_qa",
           "bugs_escaped_prod", "ci_build_failures", "story_points_committed"]
CATEGORICAL = ["department"]
FEATURE_COLS = NUMERIC + CATEGORICAL
TARGET = "completion_rate"


def build_model(input_dim):
    reg = tf.keras.regularizers.l2(1e-3)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss="mse", metrics=["mae"])
    return model


def main():
    df = pd.read_csv("data/sprints.csv")
    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    X_train_t = prep.fit_transform(X_train).astype("float32")
    X_val_t = prep.transform(X_val).astype("float32")
    X_test_t = prep.transform(X_test).astype("float32")

    model = build_model(X_train_t.shape[1])
    history = model.fit(
        X_train_t, y_train.values, validation_data=(X_val_t, y_val.values),
        epochs=250, batch_size=24, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss")],
    )

    pred = model.predict(X_test_t, verbose=0).flatten()

    # Baseline for comparison: plain linear regression on the same features
    lr = LinearRegression().fit(X_train_t, y_train.values)
    lr_pred = lr.predict(X_test_t)

    metrics = {
        "tensorflow_mlp": {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(root_mean_squared_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        },
        "linear_regression_baseline": {
            "mae": float(mean_absolute_error(y_test, lr_pred)),
            "rmse": float(root_mean_squared_error(y_test, lr_pred)),
            "r2": float(r2_score(y_test, lr_pred)),
        },
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "epochs_trained": len(history.history["loss"]),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    print(metrics["tensorflow_mlp"], "vs baseline", metrics["linear_regression_baseline"])

    joblib.dump(prep, "model/artifacts/sprint_preprocessor.joblib")
    model.save("model/artifacts/sprint_model.keras")
    with open("model/artifacts/sprint_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    bg_idx = np.random.default_rng(42).choice(len(X_train_t), size=min(100, len(X_train_t)), replace=False)
    np.save("model/artifacts/sprint_background.npy", X_train_t[bg_idx])

    sample = X_test.copy()
    sample[TARGET] = y_test.values
    sample.sample(n=min(60, len(sample)), random_state=42).to_csv(
        "model/artifacts/sprint_sample.csv", index=False
    )
    print("Saved sprint model artifacts.")


if __name__ == "__main__":
    main()
