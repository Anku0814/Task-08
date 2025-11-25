
"""
Titanic Survival Prediction with Feature Engineering (Task 08)

This script implements a full machine learning pipeline on the
Kaggle *Titanic: Machine Learning from Disaster* dataset.

Steps:
1. Load the training data (train.csv).
2. Basic EDA-style information and missing value summary.
3. Feature engineering:
   - Extract passenger title from Name.
   - Create FamilySize and IsAlone features.
   - Indicator for known Cabin.
   - Handle missing values for numeric and categorical features.
4. Encode categorical variables using OneHotEncoder.
5. Train a RandomForestClassifier model.
6. Evaluate the model (Accuracy, classification report).
7. Export feature importance plot to outputs/feature_importances.png.

Expected file structure (for GitHub repo):
.
├── data/
│   └── train.csv              # Kaggle Titanic training data
├── outputs/
│   └── feature_importances.png (generated after running)
├── titanic_feature_engineering.py
├── README.md
└── requirements.txt

Usage:
    python titanic_feature_engineering.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = os.path.join("data", "train.csv")
OUTPUT_DIR = "outputs"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Please place the Kaggle Titanic train.csv file in the data/ directory."
        )
    df = pd.read_csv(path)
    return df


def basic_info(df: pd.DataFrame) -> None:
    print("\n===== BASIC INFO =====")
    print(df.head())
    print("\nShape:", df.shape)

    print("\n===== MISSING VALUES (top 20) =====")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing.head(20))


def extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", name)
    if match:
        return match.group(1).strip()
    return "Unknown"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Title from Name
    df["Title"] = df["Name"].apply(extract_title)

    # Group rare titles
    rare_titles = [
        "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
        "Rev", "Sir", "Jonkheer", "Dona"
    ]
    df["Title"] = df["Title"].replace(
        {
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
        }
    )
    df["Title"] = df["Title"].apply(lambda x: "Rare" if x in rare_titles else x)

    # FamilySize = SibSp + Parch + 1 (self)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # IsAlone
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Cabin known flag
    df["CabinKnown"] = df["Cabin"].notnull().astype(int)

    # Drop columns that won't be used directly
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def build_model(preprocessor: ColumnTransformer) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def plot_feature_importances(pipe: Pipeline, top_n: int = 20, output_dir: str = OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    rf_model = pipe.named_steps["model"]
    preprocessor = pipe.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = rf_model.feature_importances_

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(fi["feature"], fi["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances (RandomForestClassifier)")
    output_path = os.path.join(output_dir, "feature_importances.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nFeature importance plot saved to: {output_path}")


def main():
    print("Loading data from:", DATA_PATH)
    df = load_data(DATA_PATH)

    basic_info(df)

    # Target and features
    y = df["Survived"].values
    X = df.drop(columns=["Survived"])

    # Feature engineering
    X_eng = engineer_features(X)

    preprocessor = build_preprocessor(X_eng)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_eng, y, test_size=0.2, random_state=42, stratify=y
    )

    model_pipe = build_model(preprocessor)
    print("\nTraining RandomForestClassifier model...")
    model_pipe.fit(X_train, y_train)

    y_pred = model_pipe.predict(X_valid)

    acc = accuracy_score(y_valid, y_pred)
    print("\n===== MODEL PERFORMANCE =====")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_valid, y_pred))

    plot_feature_importances(model_pipe, top_n=20, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
