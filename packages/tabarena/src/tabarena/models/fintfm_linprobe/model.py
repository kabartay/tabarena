from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, ClassVar

from tabarena.models.fintfm.checkpoints import (
    BINARY_CHECKPOINT,
    MULTICLASS_CHECKPOINT_UNPUBLISHED,
    REGRESSION_CHECKPOINT_UNPUBLISHED,
)
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
#: Keys a probe option dict accepts; see :meth:`FinTFMLinearProbeModel._validated_probe`.
PROBE_OPTIONS = frozenset({"scale", "cv", "raw", "head", "no_rep"})
#: Default of the ``probe_crossfit_folds`` hyperparameter. The measured TabArena-Lite entry
#: (FinTFM-LinProbe, 932 Elo) was produced WITHOUT cross-fitting, so any figure from a run
#: with this default is a new measurement; see :class:`FinTFMLinearProbeModel`.
PROBE_CROSSFIT_FOLDS = 5


class FinTFMLinearProbeModel(FinTFMModel):
    """fintfm, decided by a linear probe on its representation rather than its own head.

    Sibling of :class:`~tabarena.models.fintfm.model.FinTFMModel` whose default config is the
    measured TabArena-Lite entry (see ``_entry_*`` below). fintfm ``docs/results/FINDINGS.md`` §130/§131 measured that a plain
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

    **Cross-fitting (``probe_crossfit_folds``, default 5).** The rows the readout trains on
    are also rows of the stored in-context set, so a plain ``transform_representation`` call
    lets a context row attend to its own labelled copy, which a test row never can. With
    ``probe_crossfit_folds`` of K >= 2, the context rows among the readout's training rows are
    split into K folds and each fold is represented by the classifier restricted
    (``FinancialTFMClassifier.restrict_context``) to the other K - 1 folds; probe rows that are
    not in the context are represented under the full context, as test rows are. The probe
    rows' categorical encoding is then the out-of-fold encoding the classifier was fitted on,
    not the full-statistics encoding applied in-sample. ``0`` or ``1`` disables both and
    reproduces the earlier readout exactly. **The measured 932 Elo entry was produced without
    cross-fitting. The default here is on because that is the methodologically correct
    setting, and the change must be re-measured before any claim is made about it.**
    Test rows are still represented under the full context, so the readout's training rows
    come from contexts (K - 1)/K the size of the one it is applied under. The ``head`` probe
    option is not cross-fitted and is refused while cross-fitting is on; threshold regression
    (``FINTFM_REG_THRESHOLDS``) is not cross-fitted and keeps the earlier readout with a
    warning.
    """

    ag_key = "TA-FINTFM-LINPROBE"
    ag_name = "FinTFM-LinProbe"
    ag_priority = 65
    seed_name = "random_state"

    #: All three problem types, because the entry is measured on all three. The multiclass and
    #: regression checkpoints are not published yet, so a fit on either raises
    #: ``CheckpointNotPublishedError`` unless a local checkpoint is supplied.
    _supported_problem_types: ClassVar[list[str]] = ["binary", "multiclass", "regression"]

    #: The measured TabArena-Lite entry (fintfm ``docs/results/FINDINGS.md`` §232, §235, §237).
    #: The default config reproduces it with no environment variable set:
    #:
    #: - binary, run ``fintfm_lp_bin_scalecvraw_1114``: the published checkpoint, partition ``{}``
    #:   (the class defaults), probe ``scale`` + ``cv`` + ``raw``;
    #: - multiclass, run ``fintfm_lp_mc_scr_bag8_1113``: partition ``n_bags`` 8, the same probe;
    #: - regression, run ``fintfm_lp_reg_rawpart4_pure_1114``: partition ``n_bags`` 4, probe ``raw``,
    #:   10 bins.
    #:
    #: All three use ``feature_transform`` ``squash`` and ``max_features_strategy`` ``select``.
    _entry_params: ClassVar[dict] = {"feature_transform": "squash", "max_features_strategy": "select"}
    _entry_checkpoint: ClassVar[dict] = {
        "binary": BINARY_CHECKPOINT,
        "multiclass": MULTICLASS_CHECKPOINT_UNPUBLISHED,
        "regression": REGRESSION_CHECKPOINT_UNPUBLISHED,
    }
    _entry_partition: ClassVar[dict] = {"binary": {}, "multiclass": {"n_bags": 8}, "regression": {"n_bags": 4}}
    _entry_probe: ClassVar[dict] = {
        "binary": {"scale": True, "cv": True, "raw": True},
        "multiclass": {"scale": True, "cv": True, "raw": True},
        "regression": {"raw": True},
    }
    _per_problem_options: ClassVar[tuple[str, ...]] = ("checkpoint", "partition", "probe")

    #: The probe options of the current fit, set by ``_fit``.
    _probe_opts: dict = {}
    #: Cross-fit folds of the current fit (0 or 1: off), set with the probe options.
    _crossfit_folds: int = 0
    #: Whether the readout of the current fit is cross-fitted.
    _crossfit_on: bool = False
    #: The matrix the classifier was fitted on, held only while ``_fit`` builds the readout.
    _train_matrix = None
    #: Counters of the cross-fit work of the current fit, for the log line.
    _crossfit_stats: ClassVar[dict] = {}

    def _set_default_params(self):
        super()._set_default_params()
        self._set_default_param_value("probe_crossfit_folds", PROBE_CROSSFIT_FOLDS)

    def preprocess(self, X, **kwargs):
        """AutoGluon's preprocess, keeping the training matrix when the readout is cross-fitted.

        The matrix returned for ``is_train=True`` is exactly what the classifier is fitted on,
        with out-of-fold categorical encoding. Cross-fitting needs it to know which rows are in
        which leaf's fit rows and to give the readout the same encoding.
        """
        out = super().preprocess(X, **kwargs)
        if kwargs.get("is_train") and self._crossfit_folds >= 2:
            self._train_matrix = out
        return out

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
        crossfit = self._crossfit_folds >= 2
        if crossfit and hasattr(self.model, "thresholds_"):
            logger.warning(
                "FinTFM-LinProbe: probe_crossfit_folds=%d ignored, threshold regression is not cross-fitted.",
                self._crossfit_folds,
            )
            crossfit = False
        train_matrix, self._train_matrix = self._train_matrix, None
        self._crossfit_stats = {"restricted": 0, "context_rows": 0, "other_rows": 0, "skipped": 0}
        if crossfit:
            # The matrix the classifier was fitted on: out-of-fold categorical encoding, and
            # the routing the partition used at fit time.
            Xp = train_matrix
            if Xp is None:
                raise AssertionError("cross-fitting needs the training matrix, which preprocess did not keep")
        else:
            # The same preprocessing _fit just used, replayed with is_train=False so the already-
            # fitted categorical encoder is reused rather than refit on the same rows twice. The
            # public preprocess, as at predict time: it first drops the columns AutoGluon removed
            # for this model, which the private _preprocess alone keeps, and the fitted classifier
            # then refuses the wider table (Bioresponse on TabArena-Lite, 2026-11-11).
            Xp = self.preprocess(X, is_train=False)
        self._crossfit_on = crossfit
        y_arr = np.asarray(y)
        self._ridge_probe = None
        if self.problem_type == "regression":
            self._leaf_probes = None
            self._linear_probe = None
            self._fit_ridge_probe(np.asarray(Xp, dtype=np.float32), y_arr.astype(np.float64))
            self._log_crossfit()
            return
        if hasattr(self.model, "trees_"):
            # The partition setting wrapped the classifier: one probe per leaf context.
            self._fit_leaf_probes(np.asarray(Xp, dtype=np.float32), y_arr)
            self._log_crossfit()
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
        Xp = np.asarray(Xp, dtype=np.float32)
        self._fit_raw_conditioner(Xp)
        # The classifier was fitted on every row, so a row's position in its fit rows is its index.
        fit_idx = sub if crossfit else None
        self._linear_probe = self._new_classifier_probe().fit(self._probe_input(self.model, Xp[sub], fit_idx), y_sub)
        self._log_crossfit()

    def _resolve_entry(self, params: dict) -> dict:
        """The base class's entry settings, with the probe validated and kept for fit and predict.

        Also takes ``probe_crossfit_folds`` out of ``params`` so the classifier never sees it.

        Raises:
            ValueError: ``probe_crossfit_folds`` is not a non-negative integer, or is 2 or more
                with the ``head`` probe option, which is not cross-fitted.
        """
        folds = params.pop("probe_crossfit_folds", PROBE_CROSSFIT_FOLDS)
        if isinstance(folds, bool) or not isinstance(folds, int) or folds < 0:
            raise ValueError(f"probe_crossfit_folds must be a non-negative integer, got {folds!r}")
        entry = super()._resolve_entry(params)
        self._probe_opts = entry["probe"] = self._validated_probe(entry["probe"], source=f"probe_{self.problem_type}")
        if folds >= 2 and self._probe_opts.get("head"):
            raise ValueError(
                "probe option 'head' reads the network's own prediction on the readout rows, which "
                "is not cross-fitted; set probe_crossfit_folds to 0 to use it"
            )
        self._crossfit_folds = folds
        return entry

    def _log_crossfit(self) -> None:
        """Log what cross-fitting cost in this fit."""
        st = self._crossfit_stats
        if not self._crossfit_on:
            return
        logger.warning(
            "FinTFM-LinProbe: cross-fit K=%d: %d restricted context(s) built, %d context-row and %d "
            "other readout row(s) represented, %d estimator(s) left uncross-fitted.",
            self._crossfit_folds,
            st["restricted"],
            st["context_rows"],
            st["other_rows"],
            st["skipped"],
        )

    def _fit_ridge_probe(self, Xp: np.ndarray, y_arr: np.ndarray) -> None:
        """Fit a ridge readout on the binned regressor's pre-head representation.

        The regression counterpart of the classification probe. ``FinancialTFMRegressor``
        conditions a classifier on quantile-bin indices of the target; its pre-head
        representation of a row is read here and a standardised ``RidgeCV`` maps it to the
        continuous target directly, so the prediction is no longer limited to a weighted
        average of the bin representatives. Up to ``TRAIN_SAMPLE_SIZE`` training rows, drawn
        uniformly at random.

        With a partition the model is a ``PartitionedTFMRegressor``: the global
        fallback regressor gets this probe, and every leaf with its own regressor gets a ridge
        probe of its own on the training rows routed to it, read through that leaf's
        classifier, with the row budget split as in :meth:`_fit_leaf_probes`.

        Args:
            Xp: Preprocessed training features, the rows the regressor was fitted on.
            y_arr: Continuous training targets.
        """
        import numpy as np

        n = len(y_arr)
        seed = int(self.random_seed)
        self._fit_raw_conditioner(Xp)
        self._ridge_leaf_probes = None
        if hasattr(self.model, "thresholds_"):
            # Threshold regression (FINTFM_REG_THRESHOLDS): the readout reads the K threshold logits, not a
            # representation, beside the raw features when the raw option is set.
            self._ridge_probe = self._fit_threshold_ridge(Xp, y_arr, seed)
            return
        part = self.model if hasattr(self.model, "trees_") else None
        glob = part.global_estimator_ if part is not None else self.model
        self._ridge_probe = self._fit_ridge(
            glob.classifier_, Xp, y_arr, np.arange(n), TRAIN_SAMPLE_SIZE, seed, fit_rows=self._all_rows(n)
        )
        self._ridge_leaf_probes = None
        if part is None:
            return
        routes = part.route(Xp)
        self._ridge_leaf_probes = []
        for b, leaves in enumerate(part.leaf_estimators_):
            probes = {}
            for leaf, est in leaves.items():
                if est is None:
                    probes[leaf] = None
                    continue
                members = np.flatnonzero(routes[b] == leaf)
                if members.size == 0:
                    # No training row routes here once the replayed preprocessing re-encodes
                    # the categoricals; the leaf then predicts through the global readout.
                    probes[leaf] = None
                    continue
                budget = max(LEAF_MIN_SAMPLE, round(TRAIN_SAMPLE_SIZE * len(members) / n))
                probes[leaf] = self._fit_ridge(
                    est.classifier_,
                    Xp,
                    y_arr,
                    members,
                    budget,
                    seed + 1 + b * 1000 + leaf,
                    fit_rows=self._leaf_fit_rows(part, b, members, est.classifier_),
                )
            self._ridge_leaf_probes.append(probes)
        n_leaf = sum(p is not None for probes in self._ridge_leaf_probes for p in probes.values())
        logger.warning(
            "FinTFM-LinProbe: partitioned regression, %d leaf ridge probe(s) in %d bag(s).",
            n_leaf,
            len(part.leaf_estimators_),
        )

    def _threshold_input(self, X: np.ndarray) -> np.ndarray:
        """The threshold regressor's readout input: its K survival logits, plus raw features.

        Args:
            X: Preprocessed rows.

        Returns:
            ``(n, K)`` logits, widened by the squashed features when the ``raw`` option is set.
        """
        import numpy as np

        blocks = [self.model.readout_features(X)]
        cond = getattr(self, "_raw_conditioner", None)
        if cond is not None:
            blocks.append(np.nan_to_num(cond.transform(X), nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float64))
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    def _fit_threshold_ridge(self, Xp: np.ndarray, y_arr: np.ndarray, seed: int):
        """A standardised ``RidgeCV`` on :meth:`_threshold_input` of up to ``TRAIN_SAMPLE_SIZE`` rows.

        Args:
            Xp: Preprocessed training features.
            y_arr: Continuous training targets.
            seed: Seed of the row draw, the same draw :meth:`_fit_ridge` makes.

        Returns:
            The fitted pipeline.
        """
        import numpy as np
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(seed)
        n = len(y_arr)
        sub = np.sort(rng.choice(n, size=TRAIN_SAMPLE_SIZE, replace=False)) if n > TRAIN_SAMPLE_SIZE else np.arange(n)
        alphas = np.logspace(-3, 5, 17) if self._probe_opts.get("raw") else np.logspace(-3, 3, 13)
        feats = self._threshold_input(Xp[sub])
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(feats, y_arr[sub])

    def _fit_ridge(
        self,
        clf,
        Xp: np.ndarray,
        y_arr: np.ndarray,
        members: np.ndarray,
        budget: int,
        seed: int,
        fit_rows: np.ndarray | None = None,
    ):
        """A standardised ``RidgeCV`` on ``clf``'s representation of up to ``budget`` of ``members``.

        Args:
            clf: The fitted classifier inside a binned regressor (a leaf's, or the global one).
            Xp: Preprocessed training features.
            y_arr: Continuous training targets.
            members: Indices of the training rows this probe may learn from.
            budget: Most rows given to the probe, drawn uniformly at random.
            seed: Seed of that draw.
            fit_rows: Sorted indices into ``Xp`` of the rows ``clf`` was fitted on, to cross-fit
                the representation of those among the drawn rows; ``None`` for no cross-fitting.

        Returns:
            The fitted pipeline.
        """
        import numpy as np
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(seed)
        m = len(members)
        sub = members[np.sort(rng.choice(m, size=budget, replace=False))] if m > budget else members
        # A wider grid when raw columns join the representation: a 1,024-column table can need
        # far more shrinkage than the representation alone.
        alphas = np.logspace(-3, 5, 17) if self._probe_opts.get("raw") else np.logspace(-3, 3, 13)
        fit_idx = None if fit_rows is None else _positions_in(fit_rows, sub)
        rep = self._probe_input(clf, Xp[sub], fit_idx)
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(rep, y_arr[sub])

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
        self._fit_raw_conditioner(Xp)
        self._linear_probe = self._fit_probe(
            part.global_estimator_, Xp, y_arr, np.arange(n), TRAIN_SAMPLE_SIZE, seed, fit_rows=self._all_rows(n)
        )
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
                probes[leaf] = self._fit_probe(
                    est,
                    Xp,
                    y_arr,
                    members,
                    budget,
                    seed + 1 + b * 1000 + leaf,
                    fit_rows=self._leaf_fit_rows(part, b, members, est),
                )
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

    def _fit_probe(
        self,
        est,
        Xp: np.ndarray,
        y_arr: np.ndarray,
        members: np.ndarray,
        budget: int,
        seed: int,
        fit_rows: np.ndarray | None = None,
    ):
        """A logistic regression on ``est``'s pre-head representation of ``members``, or ``None``.

        Args:
            est: A fitted ``FinancialTFMClassifier`` (a leaf's, or the global fallback).
            Xp: Preprocessed training features.
            y_arr: Training labels in the original class order.
            members: Indices of the training rows this probe may learn from.
            budget: Most rows given to the probe, drawn by :func:`_probe_rows`.
            seed: Seed of that draw.
            fit_rows: Sorted indices into ``Xp`` of the rows ``est`` was fitted on, to cross-fit
                the representation of those among the drawn rows; ``None`` for no cross-fitting.

        Returns:
            The fitted probe, or ``None`` when no training row routes to the leaf or the draw
            misses a class present among ``members``; the caller then scores with ``est``'s own
            head. A leaf fitted under
            ``missing_classes="global"`` can hold fewer classes than the table; its probe
            covers the leaf's classes and the caller widens it with the global probe.
        """
        import numpy as np

        if members.size == 0:
            # A leaf context to which none of the replayed training rows routes: the replay
            # (preprocess with is_train=False) can encode categoricals differently from the
            # fit, so the tree sends them elsewhere (HR_Analytics, 2026-11-14). The caller
            # scores such a leaf with its own head, as when the draw misses a class.
            return None
        sub = members[_probe_rows(y_arr[members], budget, seed)]
        y_sub = y_arr[sub]
        if len(np.unique(y_sub)) < len(np.unique(y_arr[members])):
            return None
        fit_idx = None if fit_rows is None else _positions_in(fit_rows, sub)
        return self._new_classifier_probe().fit(self._probe_input(est, Xp[sub], fit_idx), y_sub)

    def _all_rows(self, n: int) -> np.ndarray | None:
        """Fit rows of an estimator fitted on all ``n`` rows when cross-fitting, else ``None``."""
        import numpy as np

        return np.arange(n) if self._crossfit_on else None

    def _leaf_fit_rows(self, part, bag: int, members: np.ndarray, est) -> np.ndarray | None:
        """Indices into the training matrix of the rows a leaf's estimator was fitted on.

        The partition fits a leaf on the bag's rows that its tree routes to the leaf. ``members``
        holds every training row the same tree routes there, bag or not, so the fit rows are
        their intersection with the bag.

        Args:
            part: The fitted partitioned classifier or regressor.
            bag: Bag number.
            members: Training rows routed to the leaf, sorted.
            est: The leaf's fitted classifier.

        Returns:
            Sorted indices, or ``None`` when not cross-fitting.

        Raises:
            RuntimeError: the count disagrees with the rows ``est`` stored, which means the
                matrix routed here is not the one the partition was fitted on.
        """
        import numpy as np

        if not self._crossfit_on:
            return None
        rows = np.intersect1d(members, part.bag_rows_[bag], assume_unique=True)
        stored = getattr(est, "_raw_y", None)
        if stored is not None and len(stored) != rows.size:
            raise RuntimeError(
                f"cross-fit: leaf holds {len(stored)} fitted rows but routing the training matrix "
                f"gives {rows.size}; the readout rows would be matched to the wrong context rows"
            )
        return rows

    def _restricted(self, clf, rows: np.ndarray):
        """``clf`` restricted to ``rows`` of its fit rows, whatever its ensemble setting.

        ``restrict_context`` refuses an estimator whose ``predict_proba`` averages ensemble
        members, as the default ``n_ensemble`` of 8 does. The readout reads
        ``transform_representation``, which uses the one stored context and no members, so the
        restriction is taken on a shallow copy with ``n_ensemble`` 1 and is used for
        representations only.

        Args:
            clf: A fitted ``FinancialTFMClassifier``.
            rows: Indices into the rows its ``fit`` received.

        Returns:
            The restricted copy.
        """
        shallow = copy.copy(clf)
        shallow.n_ensemble = 1
        return shallow.restrict_context(rows)

    def _crossfit_representation(self, clf, X: np.ndarray, fit_idx: np.ndarray) -> np.ndarray:
        """Pre-head representation of ``X`` with no row seeing its own labelled copy.

        Rows whose ``fit_idx`` is in ``clf``'s stored context are split into
        ``probe_crossfit_folds`` folds by a seeded shuffle of the context, and each fold is
        represented under ``clf`` restricted to the rest of the context. Other rows are
        represented under the full context, as test rows are.

        Args:
            clf: A fitted ``FinancialTFMClassifier``.
            X: Preprocessed rows.
            fit_idx: Per row, its index among the rows ``clf``'s ``fit`` received, or -1.

        Returns:
            ``(n, d)`` representations.
        """
        import numpy as np

        ctx = np.asarray(clf.context_indices_)
        fit_idx = np.asarray(fit_idx)
        stats = self._crossfit_stats
        pos = np.minimum(np.searchsorted(ctx, fit_idx), ctx.size - 1)
        in_ctx = (fit_idx >= 0) & (ctx[pos] == fit_idx)
        k = min(self._crossfit_folds, ctx.size)
        if k < 2 and in_ctx.any():
            stats["skipped"] += 1
            logger.warning("FinTFM-LinProbe: a context of %d row(s) cannot be cross-fitted.", ctx.size)
            in_ctx[:] = False
        blocks: list[tuple[np.ndarray, np.ndarray]] = []
        out = np.flatnonzero(~in_ctx)
        if out.size:
            blocks.append((out, np.asarray(clf.transform_representation(X[out], stage="pre_head"), dtype=np.float64)))
            stats["other_rows"] += int(out.size)
        if in_ctx.any():
            parts = np.array_split(np.random.default_rng(int(self.random_seed) + 7919).permutation(ctx.size), k)
            fold_of = np.empty(ctx.size, dtype=np.int64)
            for j, part in enumerate(parts):
                fold_of[part] = j
            row_fold = fold_of[pos]
            for j, part in enumerate(parts):
                rows = np.flatnonzero(in_ctx & (row_fold == j))
                if rows.size == 0:
                    continue
                twin = self._restricted(clf, np.delete(ctx, part))
                stats["restricted"] += 1
                stats["context_rows"] += int(rows.size)
                blocks.append(
                    (rows, np.asarray(twin.transform_representation(X[rows], stage="pre_head"), dtype=np.float64))
                )
        rep = np.empty((len(X), blocks[0][1].shape[1]), dtype=np.float64)
        for rows, r in blocks:
            rep[rows] = r
        return rep

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
        return probe.predict_proba(self._probe_input(est, X))

    @staticmethod
    def _validated_probe(opts, *, source: str) -> dict:
        """Check readout options; ``None`` or ``{}`` is the plain readout.

        The plain readout is a ``LogisticRegression`` at ``C=1`` on the raw representation, and
        a standardised ``RidgeCV`` for regression. The keys, each a boolean:

        - ``scale``: standardise the probe's inputs before the logistic regression.
        - ``cv``: choose the logistic regression's ``C`` by 3-fold cross-validated log loss on
          the probe rows, instead of the fixed ``C=1``.
        - ``raw``: append the row's own features to the representation, squashed (median, IQR
          and soft clip, fitted on the training rows) with missing values at the median. The
          readout can then reach a linear model of the features as well as the network's
          representation of them. Applies to the ridge readout too.
        - ``head``: append the network's own output for the row: the log of its head's class
          probabilities, or for regression the binned regressor's own prediction. The readout
          can then keep what the head gets right and correct the rest.
        - ``no_rep``: with ``raw``, drop the representation and read the squashed features
          alone, on the same leaves and rows: the matched raw-only control (fintfm FINDINGS
          §238). The network then only supplies the head for leaves whose probe is ``None``.

        Args:
            opts: The ``probe_<type>`` hyperparameter: ``None`` or a dict of the keys above.
            source: Where the value came from, for the message.

        Returns:
            The options as a dict, empty for the plain readout.

        Raises:
            ValueError: the value is not a dict, names an unknown key, or sets ``no_rep``
                without ``raw``.
        """
        if opts is None:
            return {}
        if not isinstance(opts, dict):
            raise ValueError(f"{source} must be a JSON object, got {opts!r}")
        unknown = set(opts) - PROBE_OPTIONS
        if unknown:
            raise ValueError(f"{source} has unknown key(s) {sorted(unknown)}; known: {sorted(PROBE_OPTIONS)}")
        if opts.get("no_rep") and not opts.get("raw"):
            raise ValueError(f"{source}: no_rep needs raw, the readout would have no input")
        return dict(opts)

    def _new_classifier_probe(self):
        """An unfitted classification readout, as :meth:`_probe_options` configures it.

        Returns:
            A scikit-learn classifier with ``fit`` and ``predict_proba``.
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        opts = self._probe_opts
        if opts.get("cv"):
            probe = LogisticRegressionCV(Cs=np.logspace(-3, 2, 6), cv=3, scoring="neg_log_loss", max_iter=2000)
        else:
            probe = LogisticRegression(max_iter=1000)
        return make_pipeline(StandardScaler(), probe) if opts.get("scale") else probe

    def _fit_raw_conditioner(self, Xp: np.ndarray) -> None:
        """Fit the squash applied to the raw features the ``raw`` option appends, if set.

        Args:
            Xp: Preprocessed training features, all training rows.
        """
        from fintfm.inference.preprocess import FeatureConditioner

        self._raw_conditioner = None
        if self._probe_opts.get("raw"):
            self._raw_conditioner = FeatureConditioner("squash", random_state=int(self.random_seed)).fit(Xp)

    def _probe_input(self, est, X: np.ndarray, fit_idx: np.ndarray | None = None) -> np.ndarray:
        """The readout's input for rows ``X``: ``est``'s pre-head representation, plus options.

        Args:
            est: The fitted network whose context conditions the representation.
            X: Preprocessed rows.
            fit_idx: Only when fitting a cross-fitted readout: per row, its index among the rows
                ``est`` was fitted on, or -1. ``None`` represents every row under the full
                context, as at predict time.

        Returns:
            ``(n, d)`` representation, widened by the squashed features when the ``raw``
            option is set and by the network's own output when ``head`` is set.
        """
        import numpy as np

        opts = self._probe_opts
        if opts.get("no_rep"):
            blocks = []
        elif fit_idx is not None:
            blocks = [self._crossfit_representation(est, X, fit_idx)]
        else:
            blocks = [np.asarray(est.transform_representation(X, stage="pre_head"), dtype=np.float64)]
        cond = getattr(self, "_raw_conditioner", None)
        if cond is not None:
            blocks.append(np.nan_to_num(cond.transform(X), nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float64))
        if self._probe_opts.get("head"):
            if self.problem_type == "regression":
                blocks.append(np.asarray(self.model.predict(X), dtype=np.float64).reshape(-1, 1))
            else:
                blocks.append(np.log(np.clip(np.asarray(est.predict_proba(X), dtype=np.float64), 1e-6, 1.0)))
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        import numpy as np

        if getattr(self, "_ridge_probe", None) is not None:
            # Regression: AutoGluon reads this method's output as the predictions themselves.
            Xp = np.asarray(self.preprocess(X, **kwargs), dtype=np.float32)
            if hasattr(self.model, "thresholds_"):
                return self._ridge_probe.predict(self._threshold_input(Xp))
            if getattr(self, "_ridge_leaf_probes", None) is None:
                return self._ridge_probe.predict(self._probe_input(self.model.classifier_, Xp))
            part = self.model
            routes = part.route(Xp)
            out = np.zeros(Xp.shape[0], dtype=np.float64)
            for b, leaves in enumerate(part.leaf_estimators_):
                for leaf in np.unique(routes[b]):
                    rows = np.flatnonzero(routes[b] == leaf)
                    est, probe = leaves.get(int(leaf)), self._ridge_leaf_probes[b].get(int(leaf))
                    if est is None or probe is None:
                        est, probe = part.global_estimator_, self._ridge_probe
                    out[rows] += probe.predict(self._probe_input(est.classifier_, Xp[rows]))
            return out / len(part.leaf_estimators_)
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
                    proba = self._score(est, probe, Xp[rows])
                    if proba.shape[1] < len(part.classes_):
                        # A leaf lacking some classes (missing_classes="global"): the global
                        # probe supplies their probability, the leaf's probe shares the rest.
                        classes = probe.classes_ if probe is not None else est.classes_
                        glob = self._score(part.global_estimator_, self._linear_probe, Xp[rows])
                        proba = part.combine_with_global(proba, classes, glob)
                    out[rows] += proba
            return self._convert_proba_to_unified_form(out / len(part.leaf_estimators_))
        if self._linear_probe is None:
            # Fell back to the network's own head at fit time; delegate to it on the raw
            # frame, since super()._predict_proba does its own preprocess call and would
            # double-preprocess an already-converted array otherwise.
            return super()._predict_proba(X, **kwargs)
        X = np.asarray(self.preprocess(X, **kwargs), dtype=np.float32)
        proba = self._linear_probe.predict_proba(self._probe_input(self.model, X))
        return self._convert_proba_to_unified_form(proba)


def _positions_in(fit_rows: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Position of each of ``rows`` within sorted ``fit_rows``, or -1 where it is absent.

    Args:
        fit_rows: Sorted, distinct indices of the rows an estimator was fitted on.
        rows: Indices to look up.

    Returns:
        Integer array aligned with ``rows``.
    """
    import numpy as np

    if fit_rows.size == 0:
        return np.full(len(rows), -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(fit_rows, rows), fit_rows.size - 1)
    return np.where(fit_rows[pos] == rows, pos, -1).astype(np.int64)


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
