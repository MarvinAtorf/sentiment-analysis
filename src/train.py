import argparse
import os

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def load_and_validate_data(data_path: str) -> pd.DataFrame:
    """
    Load training data from a CSV file and check required columns.
    """
    df = pd.read_csv(data_path)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns")

    return df

def split_data(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """
    Create and train the sentiment analysis pipeline.
    """
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    return model

def save_model(model: Pipeline, model_path: str) -> None:
    """
    Save the trained model to a file.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(model, model_path)

    print(f"Saved model to {model_path}")

def main(data_path: str, model_path: str) -> None:
    """
    Run the full training workflow.
    """
    df = load_and_validate_data(data_path)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.3f}")

    save_model(model, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sentiments.csv")
    parser.add_argument("--out", default="models/sentiment.joblib")

    args = parser.parse_args()
    main(data_path=args.data, model_path=args.out)
    