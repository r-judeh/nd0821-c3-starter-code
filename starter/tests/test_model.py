from sklearn.linear_model import LogisticRegression
from starter.ml.model import load_model, save_model
from starter.ml.model import train_model, compute_model_metrics, inference


def test_load_model(root_path):
    model = load_model(root_path, "model.pkl")

    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 300
    assert model.n_features_in_ == 109


def test_train_model(data):
    X, y = data
    model = train_model(X, y)

    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 300
    assert model.n_features_in_ == 4


def test_inference(model, data):
    X, y = data
    y_pred = inference(model, X)

    assert len(y_pred) == len(y)
    assert y_pred.any() == 1


def test_compute_model_metrics(model, data):
    X, y = data
    y_pred = inference(model, X)

    precision, recall, fbeta = compute_model_metrics(y, y_pred)

    assert precision > 0.5
    assert recall > 0.5
    assert fbeta > 0.5
