from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

POSTPROCESSOR_REGISTRY = {
    "msp": "src.postprocessors.base_postprocessor.BasePostprocessor",
    "odin": "src.postprocessors.odin_postprocessor.ODINPostprocessor",
    "vim": "src.postprocessors.vim_postprocessor.VIMPostprocessor",
    "dropout": "src.postprocessors.dropout_postprocessor.DropoutPostProcessor",
    "ash": "src.postprocessors.ash_postprocessor.ASHPostprocessor",
    "scale": "src.postprocessors.scale_postprocessor.ScalePostprocessor",
    "mls": "src.postprocessors.mls_postprocessor.MLSPostprocessor",
    "ebo": "src.postprocessors.ebo_postprocessor.EBOPostprocessor",
    "gen": "src.postprocessors.gen_postprocessor.GENPostprocessor",
    "knn": "src.postprocessors.knn_postprocessor.KNNPostprocessor",
    "dice": "src.postprocessors.dice_postprocessor.DICEPostprocessor",
    "residual": "src.postprocessors.residual_postprocessor.ResidualPostprocessor",
    "nnguide": "src.postprocessors.nnguide_postprocessor.NNGuidePostprocessor",
    "mds": "src.postprocessors.mds_postprocessor.MDSPostprocessor",
    "rmds": "src.postprocessors.rmds_postprocessor.RMDSPostprocessor",
}


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


@dataclass
class PostprocessorConfig:
    data: SimpleNamespace

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "PostprocessorConfig":
        return cls(_to_namespace(cfg))

    def __getattr__(self, item: str) -> Any:
        return getattr(self.data, item)


def _import_class(path: str):
    module_path, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_postprocessor(config: PostprocessorConfig):
    name = config.postprocessor.name
    if name not in POSTPROCESSOR_REGISTRY:
        raise KeyError(f"Unsupported postprocessor '{name}'.")
    cls = _import_class(POSTPROCESSOR_REGISTRY[name])
    return cls(config)
