from src.train import load_and_validate_data, split_data, train_model
from src.predection import predict_texts, format_prediction_lines


def test_prediction_with_sentiments_csv():
    df = load_and_validate_data("data/sentiments.csv")

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)

    texts = X_test.iloc[:2].tolist()

    preds, probs = predict_texts(model, texts)

    assert len(preds) == len(texts)
    assert len(probs) == len(texts)

    for pred in preds:
        assert pred in [0, 1]


def test_prediction_format_with_sentiments_csv():
    df = load_and_validate_data("data/sentiments.csv")

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)

    texts = X_test.iloc[:2].tolist()
    preds, probs = predict_texts(model, texts)

    result = format_prediction_lines(texts, preds, probs)

    assert len(result) == len(texts)

    for line in result:
        assert isinstance(line, str)
        assert "\t" in line