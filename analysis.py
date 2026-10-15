"""Compare two classifiers for the course personality questionnaire."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/Datos_pr.csv"))
    parser.add_argument("--output", type=Path, default=Path("images/model_comparison.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.data, encoding="cp1252")
    required = {"Response Id", "Personality"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    if df.isna().any().any():
        raise ValueError("The dataset contains missing values; review them before modeling")

    X = df.drop(columns=["Response Id", "Personality"])
    y = df["Personality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=2025, stratify=y
    )

    models = {
        "Clase mayoritaria": DummyClassifier(strategy="most_frequent"),
        "Árbol (prof. 15)": DecisionTreeClassifier(max_depth=15, random_state=2025),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=25,
            min_samples_split=5,
            random_state=2025,
            n_jobs=-1,
        ),
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, prediction),
                "macro_f1": f1_score(y_test, prediction, average="macro"),
            }
        )

    print(
        f"Registros: {len(df):,}; preguntas: {X.shape[1]}; clases: {y.nunique()}; "
        f"entrenamiento: {len(X_train):,}; prueba: {len(X_test):,}"
    )
    for row in results:
        print(f"{row['model']}: accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([r["model"] for r in results], [r["accuracy"] for r in results], color="#B65F34")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Exactitud en el conjunto de prueba")
    ax.set_title("Comparación de modelos · 18.000 registros de prueba")
    for i, row in enumerate(results):
        is_near_edge = row["accuracy"] > 0.90
        ax.text(
            row["accuracy"] - 0.012 if is_near_edge else row["accuracy"] + 0.012,
            i,
            f"{row['accuracy']:.1%}",
            va="center",
            ha="right" if is_near_edge else "left",
            color="white" if is_near_edge else "black",
        )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
