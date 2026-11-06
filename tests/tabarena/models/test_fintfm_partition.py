"""``FINTFM_PARTITION`` wraps the FinTFM classifier in tree-partitioned contexts (fintfm fix 4)."""

from __future__ import annotations

import numpy as np
import pytest

from tabarena.models.fintfm.model import FinTFMModel


@pytest.mark.parametrize("raw", [None, "", "  "])
def test_unset_or_blank_leaves_the_classifier_unpartitioned(monkeypatch: pytest.MonkeyPatch, raw: str | None) -> None:
    """The submitted default config never partitions."""
    monkeypatch.delenv("FINTFM_PARTITION", raising=False)
    if raw is not None:
        monkeypatch.setenv("FINTFM_PARTITION", raw)
    assert FinTFMModel._env_partition() is None


def test_empty_object_takes_the_class_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """``{}`` switches partitioning on with §169's depth 2 and 5% leaf floor."""
    monkeypatch.setenv("FINTFM_PARTITION", "{}")
    assert FinTFMModel._env_partition() == {}


def test_named_arguments_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINTFM_PARTITION", '{"max_depth": 1, "min_leaf_rows": 100}')
    assert FinTFMModel._env_partition() == {"max_depth": 1, "min_leaf_rows": 100}


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "JSON object"),
        ('{"max_detph": 2}', "max_detph"),
        ('{"random_state": 3}', "random_state"),
        ('{"estimator": null}', "estimator"),
    ],
)
def test_bad_values_raise(monkeypatch: pytest.MonkeyPatch, raw: str, match: str) -> None:
    """A misspelled or reserved key raises rather than running the default under the arm's name."""
    monkeypatch.setenv("FINTFM_PARTITION", raw)
    with pytest.raises(ValueError, match=match):
        FinTFMModel._env_partition()


class _Net:
    """Stands in for the network: records where it was moved."""

    def __init__(self) -> None:
        self.moved_to: list[str] = []

    def to(self, device: str) -> None:
        self.moved_to.append(device)


class _Clf:
    def __init__(self, net: _Net) -> None:
        self.model = net
        self.device = "mps"


class _Partitioned:
    def __init__(self, net: _Net) -> None:
        self.estimator = _Clf(net)
        self.global_estimator_ = _Clf(net)
        self.leaf_estimators_ = [{1: _Clf(net), 2: None, 3: _Clf(net)}]


def test_set_device_reaches_every_inner_classifier() -> None:
    """Every leaf, the fallback and the template follow a device change; a fallback leaf is skipped."""
    model = FinTFMModel(path="", name="fintfm_test", problem_type="binary", eval_metric="roc_auc")
    net = _Net()
    model.model = _Partitioned(net)
    model._set_device("cpu")
    inner = model._inner_classifiers()
    assert len(inner) == 4
    assert all(clf.device == "cpu" for clf in inner)
    assert np.unique(net.moved_to).tolist() == ["cpu"]
