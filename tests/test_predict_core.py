import pandas as pd
import pytest

from src.app.predict_core import validate_feature_frame, run_binary_predict, run_multiclass_predict
from src.utils.columns import ALL_FEATURES


def test_validate_feature_frame_ok():
    df = pd.DataFrame([{c: 0 for c in ALL_FEATURES}])
    df["protocol_type"] = "tcp"
    df["service"] = "http"
    df["flag"] = "SF"
    assert validate_feature_frame(df) is None


def test_validate_feature_frame_missing():
    df = pd.DataFrame([{"duration": 0}])
    err = validate_feature_frame(df)
    assert err is not None
    assert "Missing columns" in err


def test_run_binary_invalid_backend():
    df = pd.DataFrame([{c: 0 for c in ALL_FEATURES}])
    df["protocol_type"] = "tcp"
    df["service"] = "http"
    df["flag"] = "SF"
    with pytest.raises(ValueError, match="model_backend"):
        run_binary_predict(df, "xgboost")


def test_run_multiclass_invalid_backend():
    df = pd.DataFrame([{c: 0 for c in ALL_FEATURES}])
    df["protocol_type"] = "tcp"
    df["service"] = "http"
    df["flag"] = "SF"
    with pytest.raises(ValueError, match="model_backend"):
        run_multiclass_predict(df, "invalid")

# Integration tests that load artifacts are omitted here: joblib/numpy/sklearn
# versions must match training. Use `python src/test_predictions.py <api_url>` against a running API.
