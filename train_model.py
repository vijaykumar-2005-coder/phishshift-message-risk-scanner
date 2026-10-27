from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "messages.csv"
MODEL_DIR = BASE_DIR / "models"

REQUIRED_COLUMNS = {"body_text", "label"}


def main():
    # 1. Check that the dataset exists.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place messages.csv inside the data folder."
        )

    # 2. Load and validate the data.
    df = pd.read_csv(DATA_PATH)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["body_text", "label"]).copy()
    df["body_text"] = df["body_text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["body_text"] != ""]
    df = df.drop_duplicates(subset=["body_text", "label"])

    allowed_labels = {"phishing", "legitimate"}
    if not set(df["label"]).issubset(allowed_labels):
        raise ValueError(
            "Labels must be 'phishing' or 'legitimate'. "
            f"Found: {sorted(set(df['label']))}"
        )

    if df["label"].nunique() != 2 or df["label"].value_counts().min() < 2:
        raise ValueError("Need at least two examples of each label.")

    print("\nDataset loaded successfully")
    print(f"Messages: {len(df)}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    # 3. Split the data, preserving the proportion of each label.
    # This is a basic demonstration split, NOT campaign-separated evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        df["body_text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    # 4. Build and train the TF-IDF + Logistic Regression pipeline.
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=5000,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # 5. Evaluate on the held-out test set.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (labels: legitimate, phishing):")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["legitimate", "phishing"],
    ))

    # 6. Save the model and metrics.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "message_model.pkl"
    metrics_path = MODEL_DIR / "metrics.json"

    joblib.dump(model, model_path)

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "dataset_rows_after_cleaning": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy),
        "evaluation_type": "random stratified split; synthetic demonstration only",
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()