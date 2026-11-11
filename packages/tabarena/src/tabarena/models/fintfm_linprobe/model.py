from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tabarena.models.fintfm.model import FinTFMModel, prefetch_weights  # noqa: F401 (re-exported to info.py)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)

#: Rows given to the downstream classifier's training step. Mirrors
#: ``fintfm.experiments.representation_transfer.TRAIN_SAMPLE_SIZE``: every minority-class row
#: is kept, then padded with a random majority-class sample up to this budget, because a plain
#: random subsample at a low positive rate can miss the minority class outright.
TRAIN_SAMPLE_SIZE = 20_000
#: Fewest probe rows a partition leaf is given, however small its share of the training rows.
LEAF_MIN_SAMPLE = 2_000


class FinTFMLinearProbeModel(FinTFMModel):
    """fintfm, decided by a linear probe on its representation rather than its own head.

    Exploratory sibling of :class:`~tabarena.models.fintfm.model.FinTFMModel`, not a
    submission candidate. fintfm ``docs/results/FINDINGS.md`` §130/§131 measured that a plain
    ``LogisticRegression`` on the model's own pre-head representation beats its trained
    classification head on V4FinBench, 4 of 5 folds, +0.0214 mean AP, even though the probe
    reads the exact tensor the head itself reads (verified to reproduce the head's own logits
    when the head is applied to it). This class asks whether that finding transfers to
    TabArena's protocol, where the comparison is against 94 other methods on 27 datasets
    rather than two arms of one model on one binary credit task.

    Reuses every step of :class:`FinTFMModel` unchanged -- checkpoint loading, capacity
    checks, categorical preprocessing, context storage -- and changes only the decision step:
    ``_fit`` additionally extracts the pre-head representation on a bounded training
    subsample and fits a ``LogisticRegression`` on it; ``_predict_proba`` reads that
    classifier instead of the network's own head. A separate ``ag_key`` keeps its results out
    of the cache the real entrant's Lite run already populated.
    """

    ag_key = "TA-FINTFM-LINPROBE"
    ag_name = "FinTFM-LinProbe"
    ag_priority = 65
    seed_name = "random_state"

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        super()._fit(
            X,
            y,
            X_val=X_val,
            y_val=y_val,
            time_limit=time_limit,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            **kwargs,
        )
        # The same preprocessing _fit just used, replayed with is_train=False so the already-
        # fitted categorical encoder is reused rather than refit on the same rows twice. The
        # public preprocess, as at predict time: it first drops the columns AutoGluon removed
        # for this model, which the private _preprocess alone keeps, and the fitted classifier
        # then refuses the wider table (Bioresponse on TabArena-Lite, 2026-11-11).
        Xp = self.preprocess(X, is_train=False)
        y_arr = np.asarray(y)
        self._ridge_probe = None
        if self.problem_type == "regression":
            self._leaf_probes = None
            self._linear_probe = None
            self._fit_ridge_probe(np.asarray(Xp, dtype=np.float32), y_arr.astype(np.float64))
            return
        if hasattr(self.model, "trees_"):
            # FINTFM_PARTITION wrapped the classifier: one probe per leaf context.
            self._fit_leaf_probes(np.asarray(Xp, dtype=np.float32), y_arr)
            return
        self._leaf_probes = None

        sub = _probe_rows(y_arr, TRAIN_SAMPLE_SIZE, int(self.random_seed))
        y_sub = y_arr[sub]
        if len(np.unique(y_sub)) < 2:
            # A linear probe cannot be fit on one class, so fall back to the network's own head
            # for this fit rather than let one degenerate call fail the whole run. The sampling
            # in _probe_rows no longer produces a single-class subsample from a two-class y
            # (fintfm's TabArena loss analysis, D1, traced the fallbacks seen on Lite to it);
            # this guard remains for a y that arrives with one class.
            logger.warning(
                "FinTFM-LinProbe: training subsample had %d class(es), not 2; falling back "
                "to the network's own head for this fit.",
                len(np.unique(y_sub)),
            )
            self._linear_probe = None
            return
        rep = self.model.transform_representation(Xp[sub], stage="pre_head")
        self._linear_probe = LogisticRegression(max_iter=1000).fit(rep, y_sub)

    def _fit_ridge_probe(self, Xp: np.ndarray, y_arr: np.ndarray) -> None:
        """Fit a ridge readout on the binned regressor's pre-head representation.

        The regression counterpart of the classification probe. ``FinancialTFMRegressor``
        conditions a classifier on quantile-bin indices of the target; its pre-head
        representation of a row is read here and a standardised ``RidgeCV`` maps it to the
        continuous target directly, so the prediction is no longer limited to a weighted
        average of the bin representatives. Up to ``TRAIN_SAMPLE_SIZE`` training rows, drawn
        uniformly at random. Partitioned contexts are not applied to regression yet.

        Args:
            Xp: Preprocessed training features, the rows the regressor was fitted on.
            y_arr: Continuous training targets.
        """
        import numpy as np
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if self._env_partition() is not None:
            logger.warning("FinTFM-LinProbe: FINTFM_PARTITION is not applied to regression; global ridge probe only.")
        n = len(y_arr)
        rng = np.random.default_rng(int(self.random_seed))
        sub = np.sort(rng.choice(n, size=TRAIN_SAMPLE_SIZE, replace=False)) if n > TRAIN_SAMPLE_SIZE else np.arange(n)
        rep = self.model.classifier_.transform_representation(Xp[sub], stage="pre_head")
        self._ridge_probe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(rep, y_arr[sub])

    def _fit_leaf_probes(self, Xp: np.ndarray, y_arr: np.ndarray) -> None:
        """Fit one probe per leaf context of a partitioned classifier, plus one global probe.

        A leaf's representation is conditioned on that leaf's own context, so representations
        from two leaves are not in one space and a single probe across leaves would mix them.
        Each leaf with its own context therefore gets a probe trained on the training rows the
        tree routes to it, read through that leaf's classifier. The global fallback context
        gets the plain probe. The probe-row budget is split across leaves in proportion to
        their share of the training rows (at least ``LEAF_MIN_SAMPLE`` each), so the total
        representation cost stays near twice the plain probe's rather than growing with the
        number of leaves.

        Args:
            Xp: Preprocessed training features, the rows the partition was fitted on.
            y_arr: Training labels in the original class order.
        """
        import numpy as np

        part = self.model
        n = len(y_arr)
        seed = int(self.random_seed)
        self._linear_probe = self._fit_probe(part.global_estimator_, Xp, y_arr, np.arange(n), TRAIN_SAMPLE_SIZE, seed)
        routes = part.route(Xp)
        self._leaf_probes = []
        for b, leaves in enumerate(part.leaf_estimators_):
            probes = {}
            for leaf, est in leaves.items():
                if est is None:
                    probes[leaf] = None
                    continue
                members = np.flatnonzero(routes[b] == leaf)
                budget = max(LEAF_MIN_SAMPLE, round(TRAIN_SAMPLE_SIZE * len(members) / n))
                probes[leaf] = self._fit_probe(est, Xp, y_arr, members, budget, seed + 1 + b * 1000 + leaf)
            self._leaf_probes.append(probes)
        n_leaf = sum(p is not None for probes in self._leaf_probes for p in probes.values())
        n_ctx = sum(e is not None for leaves in part.leaf_estimators_ for e in leaves.values())
        logger.warning(
            "FinTFM-LinProbe: partitioned, %d leaf probe(s) over %d leaf context(s) in %d bag(s); global probe %s.",
            n_leaf,
            n_ctx,
            len(part.leaf_estimators_),
            "fitted" if self._linear_probe is not None else "absent (head fallback)",
        )

    def _fit_probe(self, est, Xp: np.ndarray, y_arr: np.ndarray, members: np.ndarray, budget: int, seed: int):
        """A logistic regression on ``est``'s pre-head representation of ``members``, or ``None``.

        Args:
            est: A fitted ``FinancialTFMClassifier`` (a leaf's, or the global fallback).
            Xp: Preprocessed training features.
            y_arr: Training labels in the original class order.
            members: Indices of the training rows this probe may learn from.
            budget: Most rows given to the probe, drawn by :func:`_probe_rows`.
            seed: Seed of that draw.

        Returns:
            The fitted probe, or ``None`` when the draw misses a class present in ``y_arr``;
            the caller then scores with ``est``'s own head.
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        sub = members[_probe_rows(y_arr[members], budget, seed)]
        y_sub = y_arr[sub]
        if len(np.unique(y_sub)) < len(np.unique(y_arr)):
            return None
        rep = est.transform_representation(Xp[sub], stage="pre_head")
        return LogisticRegression(max_iter=1000).fit(rep, y_sub)

    def _score(self, est, probe, X: np.ndarray) -> np.ndarray:
        """Class probabilities for ``X`` from ``probe`` on ``est``'s representation, or ``est``'s head.

        Args:
            est: The fitted classifier whose context conditions the representation.
            probe: Its probe, or ``None`` to use the network's own head.
            X: Preprocessed query rows.

        Returns:
            ``(n, n_classes)`` probabilities in the original class order.
        """
        if probe is None:
            proba = est.predict_proba(X)
            # The head scores the labels it was fitted on, flipped when FINTFM_MINORITY_POSITIVE
            # flipped them; the probes always see the original labels.
            return proba[:, ::-1] if getattr(self, "_minority_flipped", False) else proba
        return probe.predict_proba(est.transform_representation(X, stage="pre_head"))

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        import numpy as np

        if getattr(self, "_ridge_probe", None) is not None:
            # Regression: AutoGluon reads this method's output as the predictions themselves.
            Xp = np.asarray(self.preprocess(X, **kwargs), dtype=np.float32)
            return self._ridge_probe.predict(self.model.classifier_.transform_representation(Xp, stage="pre_head"))
        if getattr(self, "_leaf_probes", None) is not None:
            Xp = np.asarray(self.preprocess(X, **kwargs), dtype=np.float32)
            part = self.model
            routes = part.route(Xp)
            out = np.zeros((Xp.shape[0], len(part.classes_)), dtype=np.float64)
            for b, leaves in enumerate(part.leaf_estimators_):
                for leaf in np.unique(routes[b]):
                    rows = np.flatnonzero(routes[b] == leaf)
                    est, probe = leaves.get(int(leaf)), self._leaf_probes[b].get(int(leaf))
                    if est is None:
                        est, probe = part.global_estimator_, self._linear_probe
                    out[rows] += self._score(est, probe, Xp[rows])
            return self._convert_proba_to_unified_form(out / len(part.leaf_estimators_))
        if self._linear_probe is None:
            # Fell back to the network's own head at fit time; delegate to it on the raw
            # frame, since super()._predict_proba does its own preprocess call and would
            # double-preprocess an already-converted array otherwise.
            return super()._predict_proba(X, **kwargs)
        X = self.preprocess(X, **kwargs)
        rep = self.model.transform_representation(X, stage="pre_head")
        proba = self._linear_probe.predict_proba(rep)
        return self._convert_proba_to_unified_form(proba)


def _probe_rows(y_arr: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Positions in ``y_arr`` given to a probe: every minority row, padded with majority rows.

    Args:
        y_arr: Labels of the candidate rows.
        budget: Most rows returned.
        seed: Seed of the random draw.

    Returns:
        Sorted or concatenated positions into ``y_arr``, at most ``budget`` of them.
    """
    import numpy as np

    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) > 2 or (len(classes) == 2 and set(classes.tolist()) != {0, 1}):
        return _stratified_rows(y_arr, classes, counts, budget, seed)
    positive = np.where(y_arr == 1)[0]
    negative = np.where(y_arr == 0)[0]
    if len(y_arr) > budget and len(positive) >= budget:
        # Class 1 alone fills the budget (class 1 is the majority on some tasks, e.g.
        # Amazon_employee_access at 0.94). Keeping every class-1 row would leave no room
        # for class 0 and force the head fallback, so cap both classes instead:
        # each gets up to half the budget and the larger one fills what the smaller leaves.
        rng = np.random.default_rng(seed)
        n_negative = min(len(negative), budget // 2)
        n_positive = budget - n_negative
        negative = rng.choice(negative, size=n_negative, replace=False)
        positive = rng.choice(positive, size=n_positive, replace=False)
        return np.sort(np.concatenate([positive, negative]))
    if len(y_arr) > budget:
        rng = np.random.default_rng(seed)
        n_negative = budget - len(positive)
        if n_negative < len(negative):
            negative = rng.choice(negative, size=n_negative, replace=False)
        return np.concatenate([positive, negative])
    return np.arange(len(y_arr))


def _stratified_rows(y_arr: np.ndarray, classes: np.ndarray, counts: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Positions for a probe on labels other than {0, 1}: every class kept, smallest first.

    The binary rule in :func:`_probe_rows` reads class 1 as the minority and class 0 as the
    rest, so on a multiclass ``y`` it would drop every class above 1. Here the budget is
    water-filled across classes instead: classes are visited from smallest to largest, each
    taking all its rows or an equal share of what is left, whichever is smaller, so a rare
    class is kept whole and the largest classes absorb the cut.

    Args:
        y_arr: Labels of the candidate rows.
        classes: The distinct labels, from ``np.unique``.
        counts: Rows per label, aligned with ``classes``.
        budget: Most rows returned.
        seed: Seed of the random draw.

    Returns:
        Sorted positions into ``y_arr``, at most ``budget`` of them, every class present.
    """
    import numpy as np

    if len(y_arr) <= budget:
        return np.arange(len(y_arr))
    rng = np.random.default_rng(seed)
    order = np.argsort(counts, kind="stable")
    remaining = budget
    picked = []
    for i, c in enumerate(order):
        take = min(int(counts[c]), remaining // (len(order) - i))
        rows = np.flatnonzero(y_arr == classes[c])
        picked.append(rows if take >= len(rows) else rng.choice(rows, size=take, replace=False))
        remaining -= take
    return np.sort(np.concatenate(picked))
