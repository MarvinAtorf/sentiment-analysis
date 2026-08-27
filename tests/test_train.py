from src.train import load_and_validate_data, split_data, train_model


def test_real_sentiments_csv():
    df = load_and_validate_data("data/sentiments.csv")

    assert "text" in df.columns
    assert "label" in df.columns
    assert len(df) > 0


def test_split_real_data():
    df = load_and_validate_data("data/sentiments.csv")

    X_train, X_test, y_train, y_test = split_data(df)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_train_and_test_accuracy():
    df = load_and_validate_data("data/sentiments.csv")

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f"Train accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")

    assert 0.0 <= train_accuracy <= 1.0
    assert 0.0 <= test_accuracy <= 1.0