"""Single model-construction path shared by training, evaluation and inference."""

from typing import Any, Dict, Tuple

import torch

from archs import CSMambaUNeXt
from ablation_archs import (
    BaseCSMamba,
    BaseCSMambaSeaCAGhost,
    BaseCSMambaShallowCAGhost,
    BaseCSMambaShallowSea,
    BaseShallowSeaCAGhost,
    BaseUNeXt,
)


MODEL_NAME = "CSMamba-UNeXt"
BASE_NAME = "Baseline-UNeXt"
BASE_CSMAMBA_NAME = "Baseline+CSMamba"
BASE_SHALLOW_SEA_CAGHOST_NAME = "Baseline+ShallowEnhance+SEA+CA-Ghost"
BASE_CSMAMBA_SEA_CAGHOST_NAME = "Baseline+CSMamba+SEA+CA-Ghost"
BASE_CSMAMBA_SHALLOW_CAGHOST_NAME = "Baseline+CSMamba+ShallowEnhance+CA-Ghost"
BASE_CSMAMBA_SHALLOW_SEA_NAME = "Baseline+CSMamba+ShallowEnhance+SEA"
SUPPORTED_MODELS = (
    MODEL_NAME,
    BASE_NAME,
    BASE_CSMAMBA_NAME,
    BASE_SHALLOW_SEA_CAGHOST_NAME,
    BASE_CSMAMBA_SEA_CAGHOST_NAME,
    BASE_CSMAMBA_SHALLOW_CAGHOST_NAME,
    BASE_CSMAMBA_SHALLOW_SEA_NAME,
)


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    name = config.get("model_name", config.get("arch", MODEL_NAME))
    if name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {name}; choose one of {SUPPORTED_MODELS}"
        )
    input_size = int(config.get("size", config.get("input_w", 256)))
    shared = {
        "num_classes": int(config.get("num_classes", 1)),
        "img_size": input_size,
        "embed_dims": config.get("embed_dims", [128, 160, 256]),
    }
    if name in {BASE_NAME, BASE_CSMAMBA_NAME}:
        model_class = BaseUNeXt if name == BASE_NAME else BaseCSMamba
        return model_class(
            **shared,
            base_drop_rate=float(config.get("base_drop_rate", 0.0)),
            base_drop_path_rate=float(config.get("base_drop_path_rate", 0.0)),
            csmamba_drop_path_rate=float(config.get("csmamba_drop_path_rate", 0.1)),
            attention_dropout=float(config.get("attention_dropout", 0.0)),
        )
    if name == BASE_SHALLOW_SEA_CAGHOST_NAME:
        return BaseShallowSeaCAGhost(
            **shared,
            module_drop_rate=float(config.get("drop_rate", 0.2)),
            module_drop_path_rate=float(config.get("drop_path_rate", 0.1)),
        )
    if name == BASE_CSMAMBA_SEA_CAGHOST_NAME:
        return BaseCSMambaSeaCAGhost(
            **shared,
            module_drop_rate=float(config.get("drop_rate", 0.2)),
            module_drop_path_rate=float(config.get("drop_path_rate", 0.1)),
            csmamba_drop_path_rate=float(config.get("csmamba_drop_path_rate", 0.1)),
            attention_dropout=float(config.get("attention_dropout", 0.0)),
        )
    if name == BASE_CSMAMBA_SHALLOW_CAGHOST_NAME:
        return BaseCSMambaShallowCAGhost(
            **shared,
            module_drop_rate=float(config.get("drop_rate", 0.2)),
            module_drop_path_rate=float(config.get("drop_path_rate", 0.1)),
            csmamba_drop_path_rate=float(config.get("csmamba_drop_path_rate", 0.1)),
            attention_dropout=float(config.get("attention_dropout", 0.0)),
        )
    if name == BASE_CSMAMBA_SHALLOW_SEA_NAME:
        return BaseCSMambaShallowSea(
            **shared,
            base_drop_rate=float(config.get("base_drop_rate", 0.0)),
            base_drop_path_rate=float(config.get("base_drop_path_rate", 0.0)),
            csmamba_drop_path_rate=float(config.get("csmamba_drop_path_rate", 0.1)),
            attention_dropout=float(config.get("attention_dropout", 0.0)),
        )
    return CSMambaUNeXt(
        **shared,
        drop_path_rate=float(config.get("drop_path_rate", 0.1)),
        drop_rate=float(config.get("drop_rate", 0.2)),
    )


def split_outputs(outputs: Any) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Return the main logits first and the optional auxiliary logits second."""
    if isinstance(outputs, (tuple, list)):
        if not outputs:
            raise ValueError("The model returned an empty output sequence")
        main = outputs[0]
        auxiliary = outputs[1] if len(outputs) > 1 else None
        return main, auxiliary
    return outputs, None
