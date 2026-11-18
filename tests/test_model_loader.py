from pathlib import Path
import pandas as pd
import pytest
from src.features.build_features import MODEL_FEATURES
from src.models.xgboost_loader import XGBoostScorer


def test_missing_model_has_clear_status(tmp_path):
    scorer = XGBoostScorer(tmp_path / "missing.json")
    assert scorer.load() is False
    assert "unavailable" in scorer.error


def test_feature_contract_and_prediction_shape():
    model_path = Path(__file__).parents[1] / "experiments/model_results/xgboost/xgboost_baseline.json"
    if not model_path.is_file():
        pytest.skip("XGBoost artifact is not present")
    scorer = XGBoostScorer(model_path)
    if not scorer.load():
        pytest.skip(f"XGBoost runtime cannot load artifact: {scorer.error}")
    frame = pd.DataFrame([{name: 0.0 for name in MODEL_FEATURES} for _ in range(3)])
    probs = scorer.predict(frame)
    assert probs.shape == (3,)
    with pytest.raises(ValueError, match="Missing model features"):
        scorer.predict(frame.drop(columns=[MODEL_FEATURES[0]]))
