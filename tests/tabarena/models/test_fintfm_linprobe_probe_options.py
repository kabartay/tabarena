"""``FINTFM_PROBE`` configures the linear probe's readout and leaves it unchanged when unset.

The options exist to improve FinTFM's multiclass and regression entries on TabArena-Lite
(fintfm DR7.6, 2026-11-11) without changing the binary entry, so the unset case must give the
same readout as before. These tests stand in for the network with a fixed representation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fintfm")

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tabarena.models.fintfm.model import FinTFMModel
from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel


class _Network:
    """A stand-in network whose representation is two fixed columns of the row."""

    def transform_representation(self, X: np.ndarray, stage: str) -> np.ndarray:
        """Return a two-column representation of ``X``.

        Args:
            X: Query rows.
            stage: Ignored; accepted for signature compatibility.

        Returns:
            ``(n, 2)``: the first feature and a constant.
        """
        return np.column_stack([X[:, 0], np.ones(len(X))])


def _fitted(monkeypatch: pytest.MonkeyPatch, probe: str | None) -> FinTFMLinearProbeModel:
    """A probe model fitted on three features with the parent fit stubbed and ``FINTFM_PROBE`` set."""
    model = FinTFMLinearProbeModel(path="", name="fintfm_probe_opts", problem_type="multiclass", eval_metric="log_loss")
    model.random_seed = 0

    def parent_fit(self, X, y, **kwargs) -> None:
        self.model = _Network()

    monkeypatch.setattr(FinTFMModel, "_fit", parent_fit)
    monkeypatch.setattr(FinTFMLinearProbeModel, "preprocess", lambda self, X, **kw: X.to_numpy(dtype=np.float32))
    monkeypatch.delenv("FINTFM_PARTITION", raising=False)
    if probe is None:
        monkeypatch.delenv("FINTFM_PROBE", raising=False)
    else:
        monkeypatch.setenv("FINTFM_PROBE", probe)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300), "c": rng.normal(size=300)})
    X.loc[::7, "c"] = np.nan
    y = pd.Series(np.digitize(X["b"], [-0.5, 0.5]))
    model._fit(X, y)
    return model


def test_unset_keeps_the_plain_logistic_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``FINTFM_PROBE`` the probe is a bare ``C=1`` logistic regression on the representation."""
    model = _fitted(monkeypatch, None)
    assert type(model._linear_probe) is LogisticRegression
    assert model._linear_probe.n_features_in_ == 2


def test_raw_appends_the_features_and_scale_wraps_a_scaler(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raw`` widens the probe's input by the three features, missing values included."""
    model = _fitted(monkeypatch, '{"scale": true, "cv": true, "raw": true}')
    assert isinstance(model._linear_probe, Pipeline)
    assert model._linear_probe.n_features_in_ == 5
    X = np.random.default_rng(1).normal(size=(10, 3)).astype(np.float32)
    X[0, 2] = np.nan
    proba = model._linear_probe.predict_proba(model._probe_input(model.model, X))
    assert np.isfinite(proba).all()


def test_unknown_key_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misspelled option fails the fit rather than running the default readout."""
    with pytest.raises(ValueError, match="unknown key"):
        _fitted(monkeypatch, '{"sclae": true}')
