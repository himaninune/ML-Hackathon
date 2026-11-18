"""Load the selected XGBoost artifact once and score candidate features."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from src.config import MODEL_PATH
from src.features.build_features import MODEL_FEATURES


class XGBoostScorer:
    def __init__(self, path: Path = MODEL_PATH):
        self.path = Path(path)
        self.model = None
        self.error = None
        self.feature_names = list(MODEL_FEATURES)

    def load(self):
        if not self.path.is_file():
            self.error = f"XGBoost artifact unavailable: {self.path.name}"
            return False
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(str(self.path))
            names = getattr(model, "feature_names_in_", None)
            if names is None and getattr(model, "get_booster", None):
                names = model.get_booster().feature_names
            if names is not None and len(names) and list(names) != self.feature_names:
                raise ValueError("Model feature names do not match backend feature contract")
            if model.n_features_in_ != len(self.feature_names):
                raise ValueError("Model feature count does not match backend feature contract")
            self.model, self.error = model, None
            return True
        except Exception as exc:
            self.model = None
            self.error = f"Could not load XGBoost artifact ({type(exc).__name__})"
            return False

    @property
    def loaded(self):
        return self.model is not None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.loaded:
            raise RuntimeError(self.error or "XGBoost model is not loaded")
        missing = set(self.feature_names) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing model features: {sorted(missing)}")
        x = frame.loc[:, self.feature_names].replace([np.inf, -np.inf], np.nan)
        # XGBoost's native missing-value handling matches the original pipeline.
        return self.model.predict_proba(x)[:, 1]
