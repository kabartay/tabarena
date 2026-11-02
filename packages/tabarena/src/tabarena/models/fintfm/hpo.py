"""Search space for FinTFM: the inference-time levers fintfm's own measurements show to move a score.

FinTFM has no train loop, so every key is a ``FinancialTFMClassifier`` argument except
``large_n`` and its ``partition_*`` settings, which the wrapper turns into a fintfm
``PartitionedTFMClassifier`` around the classifier (``model.py``). Keys are kept only when a
fintfm measurement moved a score with them; the levers whose measured effect was nil, or that
were never measured, are left at their defaults rather than spending sampled configs on them.
Section numbers refer to fintfm ``docs/results/FINDINGS.md``.

Kept, with the measurement behind each:

- ``n_ensemble``: 8 members won 13 of 16 proxy tasks against 1 (§187, sign test p = 0.0074)
  and moved V4FinBench by +0.0058 mean AP, 3 folds up and 2 down (§196).
- ``max_context``: on TabArena-Lite, Elo 702, 745 and 756 at 500, 750 and 1,000 rows (fintfm
  ``docs/results/TABARENA.md``, audit AU5.14), so nothing below 1,000 is sampled. 8,000 rows
  was not a proxy candidate (§187), and cost grows with the square of the context.
- ``context_strategy``: ``retrieval`` won 8 proxy tasks against 1 (§187, p = 0.039) and lost
  0.186 mean AP on V4FinBench (§194), so it is sampled, at a lower weight than ``uniform``.
- ``context_positive_floor``: 0 and 16 each moved V4FinBench folds by up to 0.037 AP in both
  directions (§200); null on the mean, so it is a per-dataset choice. Uniform contexts only.
- ``large_n="partition"``: a depth-2 tree with a context per leaf gained +0.0375 mean AP on
  V4FinBench, 5 of 5 folds (§169). Depth 1 and 3 and bagged trees are untested there.
- ``finetune_steps``: 200 steps gained +0.0076 mean AP on V4FinBench, 5 of 5 folds (§163), at
  about 2,000 s of CPU per split on the proxy (§187). Only with a plain uniform context.
- ``feature_transform``: Yeo-Johnson (``power``) lost 0.0617 mean AP to ``rank`` on V4FinBench,
  5 of 5 folds (§129). A large effect on credit data, untested on general tabular data.

Left at their defaults: ``ensemble_label_swap`` and ``ensemble_transforms`` (neither a proxy
candidate on top of 8 members, §187), ``ensemble_class_shift``, ``ensemble_svd_components``
and the ``hybrid``, ``balanced`` and ``prototype`` context strategies (no measured gain).

Every key except ``n_ensemble``, ``max_context``, ``context_strategy`` and
``feature_transform`` needs a fintfm build from source newer than v0.7.1
(``context_positive_floor``, ``large_n``) or than v0.5.5 (``finetune_steps``,
``max_features_strategy``); the pinned ``fintfm==0.5.5`` extra predates them. The default
config (``manual_configs=[{}]``) uses none of them and runs on 0.5.5 unchanged.

A tuned run is GPU work: cost scales with ``n_ensemble``, ``partition_n_bags`` and the square
of ``max_context``, and fine-tuning adds gradient steps to every fit (fintfm audit AU5.13
records about 11 s per 1,000 rows on the local CPU/MPS machine at the default config).
"""

from __future__ import annotations

from tabarena.models.fintfm.model import FinTFMModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator

#: Sampled value of ``context_positive_floor`` that stands for ``None`` (no floor). ConfigSpace
#: categoricals cannot hold ``None``; the key is dropped from the config instead, so the
#: classifier's own default applies.
NO_POSITIVE_FLOOR = -1

#: Steps of dataset-specific fine-tuning when it is sampled: §163's setting, the only one measured.
FINETUNE_STEPS = 200


def build_search_space(seed: int = 1234):
    """The FinTFM search space as a ConfigSpace ``ConfigurationSpace``.

    Conditions keep conditional keys out of configs where they mean nothing, and forbidden
    clauses remove the combinations fintfm refuses or that no measurement covers, so no sampled
    config fails at fit time on a table within the checkpoint's 136 features:

    - ``context_positive_floor`` exists only under ``context_strategy="uniform"``, which is the
      only strategy the classifier accepts it with.
    - ``partition_max_depth`` and ``partition_n_bags`` exist only under ``large_n="partition"``.
    - Fine-tuning runs only with a uniform context, no positive floor and no partition. fintfm
      warns on the first two (the tuned weights would read a context unlike the one they were
      tuned on) and the third would fine-tune once per leaf, which nothing has measured.
    - ``max_features_strategy`` is ``"select"`` with fine-tuning and ``"subsample"`` without.
      Both only act on a table wider than 136 features, where fine-tuning refuses the column
      subsets that ``"subsample"`` scores, so the three wide TabArena v0.1 binary tasks
      (APSFailure, kddcup09_appetency, Bioresponse) get a result from every config.

    Args:
        seed: The ConfigSpace sampling seed, fixed so the sampled configs are reproducible.

    Returns:
        The ``ConfigurationSpace``.
    """
    from ConfigSpace import (
        Categorical,
        ConfigurationSpace,
        EqualsCondition,
        ForbiddenAndConjunction,
        ForbiddenEqualsClause,
        ForbiddenInClause,
    )

    n_ensemble = Categorical("n_ensemble", [4, 8, 16], default=8)
    max_context = Categorical("max_context", [1000, 2000], default=1000)
    context_strategy = Categorical("context_strategy", ["uniform", "retrieval"], weights=[0.75, 0.25])
    context_positive_floor = Categorical("context_positive_floor", [NO_POSITIVE_FLOOR, 0, 16])
    feature_transform = Categorical("feature_transform", ["rank", "power"], weights=[0.75, 0.25])
    large_n = Categorical("large_n", ["none", "partition"])
    partition_max_depth = Categorical("partition_max_depth", [1, 2, 3], default=2)
    partition_n_bags = Categorical("partition_n_bags", [1, 4], default=1)
    finetune_steps = Categorical("finetune_steps", [0, FINETUNE_STEPS], weights=[0.75, 0.25])
    max_features_strategy = Categorical("max_features_strategy", ["subsample", "select"])

    space = ConfigurationSpace(seed=seed)
    space.add(
        [
            n_ensemble,
            max_context,
            context_strategy,
            context_positive_floor,
            feature_transform,
            large_n,
            partition_max_depth,
            partition_n_bags,
            finetune_steps,
            max_features_strategy,
        ]
    )
    space.add(
        [
            EqualsCondition(context_positive_floor, context_strategy, "uniform"),
            EqualsCondition(partition_max_depth, large_n, "partition"),
            EqualsCondition(partition_n_bags, large_n, "partition"),
        ]
    )
    tuned = ForbiddenEqualsClause(finetune_steps, FINETUNE_STEPS)
    space.add(
        [
            ForbiddenAndConjunction(tuned, ForbiddenEqualsClause(context_strategy, "retrieval")),
            ForbiddenAndConjunction(tuned, ForbiddenInClause(context_positive_floor, [0, 16])),
            ForbiddenAndConjunction(tuned, ForbiddenEqualsClause(large_n, "partition")),
            ForbiddenAndConjunction(tuned, ForbiddenEqualsClause(max_features_strategy, "subsample")),
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(finetune_steps, 0),
                ForbiddenEqualsClause(max_features_strategy, "select"),
            ),
        ]
    )
    return space


def generate_configs_fintfm(num_random_configs: int = 200) -> list[dict]:
    """Sample FinTFM configs as ``FinTFMModel`` hyperparameter dicts.

    Args:
        num_random_configs: How many configs to sample.

    Returns:
        One dict per config with native Python values. A sampled ``context_positive_floor``
        of :data:`NO_POSITIVE_FLOOR` is dropped, and so is ``finetune_steps=0``, so each
        falls back to the classifier's default.
    """
    space = build_search_space()
    # A size of one returns a bare configuration rather than a list of one.
    configs = (
        [space.sample_configuration()] if num_random_configs == 1 else space.sample_configuration(num_random_configs)
    )
    out = []
    for config in configs:
        config = convert_numpy_dtypes(dict(config))
        if config.get("context_positive_floor") == NO_POSITIVE_FLOOR:
            del config["context_positive_floor"]
        if config.get("finetune_steps") == 0:
            del config["finetune_steps"]
        out.append(config)
    return out


gen_fintfm = CustomAGConfigGenerator(
    model_cls=FinTFMModel,
    search_space_func=generate_configs_fintfm,
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_fintfm.generate_all_bag_experiments(num_random_configs=4),
        ),
    )
