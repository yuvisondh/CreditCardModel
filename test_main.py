import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # non-interactive backend so tests don't open windows

from main import build_model, evaluate_model, load_data, preprocess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Synthetic dataset that mirrors the structure of creditcard.csv."""
    np.random.seed(42)
    n = 300
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Time"] = np.random.uniform(0, 172_792, n)
    data["Amount"] = np.random.uniform(0, 25_691, n)
    labels = np.zeros(n, dtype=int)
    labels[:20] = 1  # ~6.7 % fraud to mirror the real imbalance
    np.random.shuffle(labels)
    data["Class"] = labels
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

def test_load_data_valid(tmp_path, sample_df):
    path = tmp_path / "good.csv"
    sample_df.to_csv(path, index=False)
    df = load_data(str(path))
    assert df.shape == sample_df.shape


def test_load_data_raises_on_missing_values(tmp_path, sample_df):
    sample_df.loc[0, "Amount"] = np.nan
    path = tmp_path / "bad.csv"
    sample_df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing values"):
        load_data(str(path))


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

def test_preprocess_output_shapes(sample_df):
    X_train, X_test, y_train, y_test, _ = preprocess(sample_df)
    n_features = sample_df.shape[1] - 1  # exclude Class column
    assert X_train.shape[1] == n_features
    assert X_test.shape[1] == n_features
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]


def test_preprocess_smote_balances_classes(sample_df):
    _, _, y_train, _, _ = preprocess(sample_df)
    unique, counts = np.unique(y_train, return_counts=True)
    # After SMOTE both classes should be equal
    assert counts[0] == counts[1]


def test_preprocess_scaling(sample_df):
    X_train, _, _, _, _ = preprocess(sample_df)
    # StandardScaler on Time/Amount should yield approx zero mean after SMOTE
    assert abs(X_train["Time"].mean()) < 1.0
    assert abs(X_train["Amount"].mean()) < 1.0


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def test_build_model_output_shape():
    model = build_model(input_dim=30)
    assert model.output_shape == (None, 1)


def test_build_model_is_compiled():
    model = build_model(input_dim=30)
    # A compiled model exposes its optimizer
    assert model.optimizer is not None


def test_build_model_dense_layers():
    from tensorflow.keras.layers import Dense
    model = build_model(input_dim=30)
    dense_layers = [l for l in model.layers if isinstance(l, Dense)]
    assert len(dense_layers) == 4  # 64 → 32 → 16 → 1


# ---------------------------------------------------------------------------
# evaluate_model (end-to-end pipeline smoke test)
# ---------------------------------------------------------------------------

def test_evaluate_model_runs_without_error(sample_df):
    X_train, X_test, y_train, y_test, _ = preprocess(sample_df)
    model = build_model(input_dim=X_train.shape[1])
    model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=0)
    # Should complete without raising any exception
    evaluate_model(model, X_test, y_test, threshold=0.5)
