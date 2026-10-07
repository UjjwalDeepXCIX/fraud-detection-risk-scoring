"""SHAP-based explainability for the fraud model.

Used both offline (global feature-importance summary plot generated during
training) and online (per-prediction top contributing factors returned by
the API).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


class FraudExplainer:
    """Wraps a fitted (preprocessor + XGBoost) pipeline with a SHAP TreeExplainer.

    SHAP operates on the *transformed* feature space (post one-hot-encoding),
    so this class owns the preprocessor -> explainer hand-off and maps values
    back to human-readable feature names.
    """

    def __init__(self, pipeline: Pipeline):
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        self._explainer = shap.TreeExplainer(self.classifier)

    def shap_values(self, X_raw: pd.DataFrame) -> np.ndarray:
        X_transformed = self.preprocessor.transform(X_raw)
        return self._explainer.shap_values(X_transformed)

    def top_factors(self, X_raw: pd.DataFrame, top_k: int = 5) -> list[dict]:
        """Return the top-k contributing factors for a single-row input,
        ranked by absolute SHAP value, as JSON-serialisable dicts.
        """
        values = self.shap_values(X_raw)
        row = values[0] if values.ndim == 2 else values
        order = np.argsort(-np.abs(row))[:top_k]
        return [
            {"feature": self.feature_names[i], "impact": round(float(row[i]), 4)}
            for i in order
        ]

    def save_summary_plot(self, X_raw: pd.DataFrame, output_path: Path, max_display: int = 15) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X_transformed = self.preprocessor.transform(X_raw)
        values = self._explainer.shap_values(X_transformed)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure()
        shap.summary_plot(
            values,
            X_transformed,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
