from __future__ import annotations

from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel
from tabarena.utils.config_utils import ConfigGenerator

#: Empty for the same reason as fintfm's own hpo.py: this variant reuses fintfm's stored
#: context and pretrained weights unchanged, and adds no inference-time knob of its own.
search_space: dict = {}

gen_fintfm_linprobe = ConfigGenerator(
    model_cls=FinTFMLinearProbeModel,
    search_space=search_space,
    manual_configs=[{}],
)
