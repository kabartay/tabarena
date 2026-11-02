"""FinTFM's search space samples only configs that fintfm accepts, and the wrapper routes them."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tabarena.models.fintfm.hpo import (
    FINETUNE_STEPS,
    NO_POSITIVE_FLOOR,
    build_search_space,
    gen_fintfm,
    generate_configs_fintfm,
)
from tabarena.models.fintfm.model import FinTFMModel, _inner_estimators, _wrap_large_n

#: Enough draws that every categorical value and every allowed combination appears.
N_CONFIGS = 200

#: Keys the wrapper consumes itself; every other key is a ``FinancialTFMClassifier`` argument.
WRAPPER_KEYS = {"large_n", "partition_max_depth", "partition_n_bags"}


@pytest.fixture(scope="module")
def configs() -> list[dict]:
    """A fixed sample of the search space."""
    return generate_configs_fintfm(N_CONFIGS)


def test_sampling_is_reproducible() -> None:
    """The space is seeded, so two samples of the same size are identical."""
    assert generate_configs_fintfm(10) == generate_configs_fintfm(10)


def test_one_config_is_returned_as_a_list() -> None:
    """ConfigSpace returns a bare configuration for a sample of one; the generator does not."""
    assert len(generate_configs_fintfm(1)) == 1


def test_values_are_native_python(configs: list[dict]) -> None:
    """No numpy scalar reaches a config, where it would break YAML serialisation."""
    for config in configs:
        for value in config.values():
            assert type(value) in (int, str), (config, type(value))


def test_conditional_keys_appear_only_under_their_parent(configs: list[dict]) -> None:
    """A floor needs a uniform context, partition settings need a partition, and the floor's
    sentinel never reaches a config.
    """
    for config in configs:
        if "context_positive_floor" in config:
            assert config["context_strategy"] == "uniform"
            assert config["context_positive_floor"] in (0, 16)
        partition_keys = {k for k in config if k.startswith("partition_")}
        assert bool(partition_keys) == (config["large_n"] == "partition"), config
        assert config.get("context_positive_floor") != NO_POSITIVE_FLOOR


def test_finetuning_runs_only_where_it_was_measured(configs: list[dict]) -> None:
    """Fine-tuning is sampled only with a plain uniform context and column selection; every
    other config subsamples columns on a wide table.
    """
    tuned = [c for c in configs if "finetune_steps" in c]
    assert tuned, "the sample never fine-tunes"
    for config in tuned:
        assert config["finetune_steps"] == FINETUNE_STEPS
        assert config["context_strategy"] == "uniform"
        assert "context_positive_floor" not in config
        assert config["large_n"] == "none"
        assert config["max_features_strategy"] == "select"
    for config in configs:
        if "finetune_steps" not in config:
            assert config["max_features_strategy"] == "subsample"


def test_every_level_is_sampled(configs: list[dict]) -> None:
    """Each declared value of each key shows up in a sample of this size."""
    space = build_search_space()
    for name in space:
        seen = {c[name] for c in configs if name in c}
        declared = set(space[name].choices)
        if name == "context_positive_floor":
            declared.discard(NO_POSITIVE_FLOOR)
        if name == "finetune_steps":
            declared.discard(0)
        assert declared <= seen, (name, declared - seen)


def test_generator_puts_the_default_first() -> None:
    """The default config comes first and stays empty, so it runs on the pinned fintfm."""
    lst = gen_fintfm.generate_all_configs_lst(num_random_configs=3)
    assert len(lst) == 4
    assert {k: v for k, v in lst[0].items() if k != "ag_args"} == {}
    assert lst[0]["ag_args"]["name_suffix"] == "_c1"
    assert [c["ag_args"]["name_suffix"] for c in lst[1:]] == ["_r1", "_r2", "_r3"]


def test_every_sampled_config_passes_fintfms_own_checks(configs: list[dict]) -> None:
    """Every classifier key is a real argument, and fintfm's settings checks accept each config."""
    pytest.importorskip("fintfm")
    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.inference.large_n import PartitionedTFMClassifier

    accepted = set(inspect.signature(FinancialTFMClassifier.__init__).parameters) - {"self", "model"}
    for config in configs:
        clf_params = {k: v for k, v in config.items() if k not in WRAPPER_KEYS}
        assert set(clf_params) <= accepted, set(clf_params) - accepted
        estimator = FinancialTFMClassifier(None, **clf_params)  # the network is not read here
        estimator._check_ensemble_settings()
        partition = {k.removeprefix("partition_"): v for k, v in config.items() if k.startswith("partition_")}
        wrapped = _wrap_large_n(estimator, config["large_n"], partition)
        if isinstance(wrapped, PartitionedTFMClassifier):
            wrapped._check_settings()


def test_wrap_large_n() -> None:
    """``none`` is the classifier itself; ``partition`` wraps it; anything else is refused."""
    pytest.importorskip("fintfm")
    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.inference.large_n import PartitionedTFMClassifier

    estimator = FinancialTFMClassifier(None, random_state=7)
    assert _wrap_large_n(estimator, "none", {}) is estimator
    wrapped = _wrap_large_n(estimator, "partition", {"max_depth": 3, "n_bags": 4})
    assert isinstance(wrapped, PartitionedTFMClassifier)
    assert (wrapped.estimator, wrapped.max_depth, wrapped.n_bags, wrapped.random_state) == (estimator, 3, 4, 7)
    with pytest.raises(ValueError, match="need large_n='partition'"):
        _wrap_large_n(estimator, "none", {"max_depth": 2})
    with pytest.raises(ValueError, match="must be 'none' or 'partition'"):
        _wrap_large_n(estimator, "boost", {})


def test_set_device_moves_a_fine_tuned_copy_and_every_leaf() -> None:
    """A fine-tuned classifier predicts through ``model_``, a copy, and a partition holds one
    classifier per leaf; moving the device reaches all of them.
    """

    def estimator(tuned: bool) -> SimpleNamespace:
        return SimpleNamespace(model=MagicMock(), model_=MagicMock() if tuned else None, device="mps")

    plain, tuned, leaf = estimator(False), estimator(True), estimator(True)
    partition = SimpleNamespace(global_estimator_=plain, leaf_estimators_=[{1: tuned, 2: None}, {3: leaf}])
    assert list(_inner_estimators(partition)) == [plain, tuned, leaf]
    FinTFMModel._set_device(SimpleNamespace(model=partition), "cpu")
    for est in (plain, tuned, leaf):
        est.model.to.assert_called_once_with("cpu")
        assert est.device == "cpu"
    for est in (tuned, leaf):
        est.model_.to.assert_called_once_with("cpu")
    single = estimator(True)
    assert list(_inner_estimators(single)) == [single]
