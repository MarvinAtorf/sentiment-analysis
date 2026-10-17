# Sentiment Analysis

A small end-to-end machine learning project that classifies short texts as positive or negative. The focus is on a clean, reproducible workflow: training script, command-line prediction, automated tests, CI pipeline and a Docker image.

## What it does

- Trains a text classifier (TF-IDF features + logistic regression) on a CSV file with `text` and `label` columns.
- Predicts the sentiment of one or more texts from the command line and prints the label and the probability of the positive class.
- Runs in Docker, so predictions work without a local Python setup.

Labels: `1` = positive, `0` = negative.

## Quick start

### Option 1: Python venv

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option 2: Conda

```bash
conda create -n sentiment-env python=3.11 -y
conda activate sentiment-env
pip install -r requirements.txt
```

### Train

```bash
python src/train.py --data data/sentiments.csv --out models/sentiment.joblib
```

The script splits the data 80/20 (stratified), trains the model, prints the test accuracy and saves the model.

### Predict

```bash
python src/prediction.py "I loved this film" "That was bad"
```

Each line of the output contains the predicted label, the probability of the positive class and the input text, separated by tabs.

## Docker

The image contains the prediction script and the trained model, so it is meant for inference only.

```bash
docker build -t sentiment-analysis .
docker run --rm sentiment-analysis "I loved this film" "That was bad"
```

## Project structure

```
src/
  train.py          Data loading, split, training and saving
  prediction.py     Loading the model and scoring texts
tests/              Unit tests for training and prediction
data/sentiments.csv Training data (text, label)
models/             Trained model
Dockerfile          Inference image
.github/workflows/  CI pipeline
```

## Continuous integration

A GitHub Actions workflow runs on every push and pull request that touches code, data or dependencies:

- **Train and predict:** trains the model and runs the prediction script as a smoke test
- **Tests:** runs the test suite with pytest
- **Lint:** checks code with Ruff (lint and formatting)
- **Failure handling:** opens a GitHub issue if the training or prediction step fails on a push

## Tech stack

Python 3.11, scikit-learn, pandas, joblib, pytest, Ruff, Docker, GitHub Actions

## Limitations

- The training set is a small demo dataset (about 20 examples). Accuracy figures on it are not meaningful, and the model will not generalise to real-world text.
- The model is English-only and uses a basic bag-of-words approach, so it does not handle negation or irony well.
- The goal of the project is the engineering workflow. For a real use case, replace the data with a larger labelled dataset and evaluate the model properly.
