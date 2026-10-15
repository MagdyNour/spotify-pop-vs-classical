
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "SpotifyFeatures.csv"
FIGURES_PATH = ROOT / "figures"
FIGURES_PATH.mkdir(exist_ok=True)

FEATURES = [
    "popularity", "acousticness", "danceability", "duration_ms",
    "energy", "instrumentalness", "liveness", "loudness",
    "speechiness", "tempo", "valence",
]

def main():
    print("Loading Spotify dataset...")
    df = pd.read_csv(DATA_PATH)

    # Keep only the two target genres.
    df = df[df["genre"].isin(["Pop", "Classical"])].copy()
    df = df.dropna(subset=FEATURES + ["genre"])

    X = df[FEATURES]
    y = df["genre"]

    print(f"\nTracks: {len(df):,}")
    print("Class distribution:")
    print(y.value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Probability of the positive class, Classical.
        classical_index = list(model.classes_).index("Classical")
        probabilities = model.predict_proba(X_test)[:, classical_index]

        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(
            (y_test == "Classical").astype(int),
            probabilities,
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "ROC-AUC": auc,
        })

        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC:  {auc:.4f}")
        print(classification_report(y_test, predictions, digits=3))

        if name == "Random Forest":
            ConfusionMatrixDisplay(
                confusion_matrix(y_test, predictions, labels=["Classical", "Pop"]),
                display_labels=["Classical", "Pop"],
            ).plot(cmap="Blues", values_format="d")
            plt.title("Random Forest — Confusion Matrix")
            plt.tight_layout()
            plt.savefig(FIGURES_PATH / "confusion-matrix.png", dpi=160)
            plt.close()

            importances = pd.Series(
                model.feature_importances_,
                index=FEATURES,
            ).sort_values()

            importances.plot(kind="barh", figsize=(8, 6))
            plt.title("Random Forest — Audio Feature Importance")
            plt.xlabel("Feature importance")
            plt.tight_layout()
            plt.savefig(FIGURES_PATH / "feature-importance.png", dpi=160)
            plt.close()

    results_df = pd.DataFrame(results).sort_values(
        "ROC-AUC", ascending=False
    )
    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    results_df.to_csv(ROOT / "model-results.csv", index=False)
    print("\nSaved figures and model-results.csv.")

if __name__ == "__main__":
    main()
