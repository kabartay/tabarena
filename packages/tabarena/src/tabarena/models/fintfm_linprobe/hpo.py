from __future__ import annotations

from autogluon.common.space import Categorical

from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel
from tabarena.utils.config_utils import ConfigGenerator

#: PROPOSED search space for "FinTFM with a trained linear readout", not a tuned one. No
#: sampled config has been run on TabArena; the keys are the readout's own knobs that the
#: measured runs (fintfm ``docs/results/TABARENA.md``, "The 51-dataset Lite entry", §220 to
#: §224) switched on or off, so a sampled config stays inside what was measured:
#:
#: - ``probe_scale``, ``probe_cv``, ``probe_raw``, ``probe_head``: the ``FINTFM_PROBE`` options
#:   ``scale`` (standardise the readout's inputs), ``cv`` (choose the logistic regression's
#:   ``C`` by 3-fold log loss over 10^-3 to 10^2, the only readout regularisation exposed),
#:   ``raw`` (append the row's own squashed features, which the entry must declare) and
#:   ``head`` (append the network's own log probabilities; measured and not kept in §224).
#: - ``partition_n_bags``: bags of the tree partition (``FINTFM_PARTITION`` ``n_bags``); 1 is the
#:   binary entry, 4 the multiclass one. Not applied to regression.
#:
#: Left out on purpose: ``no_rep`` (a control, not a candidate), the partition depth and leaf
#: floor (only depth 2 and 0.05 were measured), ``max_context`` (4,000 changed nothing on
#: multiclass, §224) and 8 bags (queued, no result). The ridge readout for regression ignores
#: ``probe_scale`` and ``probe_cv``; it reads ``probe_raw`` and ``probe_head``.
#:
#: The default config is ``manual_configs=[{}]``: it carries no override, so it is whatever
#: the wrapper's own defaults are, which must equal the measured entry. These key names are
#: provisional until they match the hyperparameters ``model.py`` reads in place of the
#: ``FINTFM_PROBE`` and ``FINTFM_PARTITION`` variables; a key the wrapper does not read must
#: not be sampled. The readout adds a fit of up to 20,000 representation rows per leaf, so a
#: tuned run is GPU work, as for FinTFM.
search_space = {
    "probe_scale": Categorical(False, True),
    "probe_cv": Categorical(False, True),
    "probe_raw": Categorical(False, True),
    "probe_head": Categorical(False, True),
    "partition_n_bags": Categorical(1, 4),
}

gen_fintfm_linprobe = ConfigGenerator(
    model_cls=FinTFMLinearProbeModel,
    search_space=search_space,
    manual_configs=[{}],
)
