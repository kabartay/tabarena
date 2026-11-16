"""``probe_ensemble`` averages readout members fitted on one network; unset it changes nothing.

Synthetic data and an untrained small network, so the file runs in seconds. Numbers here are
SMOKE-TEST only; the exact reproduction against stored TabArena runs is a separate check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fintfm")
torch = pytest.importorskip("torch")

from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM, ModelConfig

from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel

N_TRAIN = 200
N_TEST = 40
MAX_CONTEXT = 80
PROBE = {"scale": True, "cv": True, "raw": True}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Unset every FINTFM_* variable the model reads, and force the CPU."""
    for name in (
        "FINTFM_PARTITION",
        "FINTFM_PROBE",
        "FINTFM_PARAMS",
        "FINTFM_MAX_CONTEXT",
        "FINTFM_CHECKPOINT",
        "FINTFM_PROBE_ENSEMBLE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FINTFM_DEVICE", "cpu")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    """An untrained network small enough to fit and score in a second."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    cfg = ModelConfig(
        max_features=8, max_classes=10, d_cell=8, d_model=16, n_layers=1, n_col_layers=1, n_heads=2, d_ff=32
    )
    FinancialTFM(cfg).save(str(path))
    return str(path)


def _data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """A binary table with a categorical column, so the cross-fit and replay matrices differ."""
    rng = np.random.default_rng(0)
    n = N_TRAIN + N_TEST
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    X["cat"] = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    y = pd.Series(((X["f0"] + 0.5 * X["f1"] + rng.normal(scale=0.5, size=n)) > 0).astype(int).to_numpy())
    return X.iloc[:N_TRAIN].reset_index(drop=True), y.iloc[:N_TRAIN], X.iloc[N_TRAIN:]


def _model(tmp_path, checkpoint: str, **hp) -> FinTFMLinearProbeModel:
    """An unfitted binary model with the partition ``{}`` entry and extra hyperparameters ``hp``."""
    params = {
        "checkpoint_binary": checkpoint,
        "max_context": MAX_CONTEXT,
        "feature_transform": "squash",
        "max_features_strategy": "select",
        "partition_binary": {},
        "probe_binary": dict(PROBE),
    }
    params.update(hp)
    return FinTFMLinearProbeModel(
        path=str(tmp_path), name="probe_ensemble", problem_type="binary", eval_metric="roc_auc", hyperparameters=params
    )


def _fit(tmp_path, checkpoint: str, **hp) -> FinTFMLinearProbeModel:
    model = _model(tmp_path, checkpoint, **hp)
    X, y, _ = _data()
    model.fit(X=X, y=y)
    return model


def _test_proba(model: FinTFMLinearProbeModel) -> np.ndarray:
    return np.asarray(model.predict_proba(_data()[2]), dtype=np.float64)


def test_default_is_the_single_readout(tmp_path, checkpoint):
    model = _fit(tmp_path, checkpoint)
    assert model._probe_ensemble is None
    assert model._ensemble_members is None
    assert model._get_model_params()["probe_ensemble"] is None


def test_env_override_sets_the_hyperparameter(monkeypatch, tmp_path, checkpoint):
    monkeypatch.setenv("FINTFM_PROBE_ENSEMBLE", "xfit, noxfit,rawonly")
    model = _model(tmp_path, checkpoint)
    model._init_params()
    # The test model passes probe hyperparameters explicitly; the override reaches the defaults.
    assert FinTFMLinearProbeModel._env_entry_overrides()["probe_ensemble"] == ("xfit", "noxfit", "rawonly")
    monkeypatch.setenv("FINTFM_PROBE_ENSEMBLE", "xfit")
    with pytest.raises(ValueError, match="FINTFM_PROBE_ENSEMBLE"):
        FinTFMLinearProbeModel._env_entry_overrides()


def test_ensemble_is_the_mean_of_the_members_fitted_separately(tmp_path, checkpoint):
    xfit = _test_proba(_fit(tmp_path / "a", checkpoint))
    noxfit = _test_proba(_fit(tmp_path / "b", checkpoint, probe_crossfit_folds=0))
    rawonly = _test_proba(
        _fit(tmp_path / "c", checkpoint, probe_binary={"scale": True, "cv": True, "raw": True, "no_rep": True})
    )
    ens = _fit(tmp_path / "d", checkpoint, probe_ensemble=["xfit", "noxfit", "rawonly"])
    assert list(ens._ensemble_members) == ["xfit", "noxfit", "rawonly"]
    np.testing.assert_allclose(_test_proba(ens), (xfit + noxfit + rawonly) / 3, rtol=0, atol=1e-9)
    two = _test_proba(_fit(tmp_path / "e", checkpoint, probe_ensemble="xfit,noxfit"))
    np.testing.assert_allclose(two, (xfit + noxfit) / 2, rtol=0, atol=1e-9)


def test_full_context_representation_is_computed_once_per_batch(tmp_path, checkpoint, monkeypatch):
    ens = _fit(tmp_path, checkpoint, probe_ensemble="xfit,noxfit,rawonly")
    calls: list[int] = []
    original = FinancialTFMClassifier.transform_representation

    def spy(clf, X, stage="encode_rows"):
        calls.append(len(X))
        return original(clf, X, stage=stage)

    monkeypatch.setattr(FinancialTFMClassifier, "transform_representation", spy)
    _test_proba(ens)
    assert calls == [N_TEST], calls  # xfit and noxfit share one pass; rawonly reads none


@pytest.mark.parametrize("value", [["xfit", "bogus"], ["xfit", "xfit"], ["xfit"], "xfit,,nope", 3])
def test_bad_member_lists_are_rejected(value, tmp_path, checkpoint):
    with pytest.raises(ValueError, match="probe_ensemble"):
        _fit(tmp_path, checkpoint, probe_ensemble=value)


def test_ensemble_needs_cross_fit_folds_and_no_prior_no_rep(tmp_path, checkpoint):
    with pytest.raises(ValueError, match="probe_crossfit_folds"):
        _fit(tmp_path / "a", checkpoint, probe_ensemble="xfit,noxfit", probe_crossfit_folds=0)
    with pytest.raises(ValueError, match="no_rep"):
        _fit(
            tmp_path / "b",
            checkpoint,
            probe_ensemble="xfit,noxfit",
            probe_binary={"raw": True, "no_rep": True},
        )
