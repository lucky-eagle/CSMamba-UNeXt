"""Documented defaults for the CSMamba-UNeXt manuscript experiments.

Runtime configuration is resolved by ``experiment.py`` and saved with every
run.  This module replaces the unrelated Swin-Transformer configuration that
was present in the original archive.
"""

MANUSCRIPT_DEFAULTS = {
    "input_size": 256,
    "batch_size": 4,
    "num_workers": 0,
    "epochs": 200,
    "optimizer": "AdamW",
    "learning_rate": 5e-4,
    "betas": (0.9, 0.999),
    "weight_decay": 1e-4,
    "scheduler": "CosineAnnealingLR",
    "t_max": 200,
    "minimum_learning_rate": 1e-5,
    "warmup_epochs": 0,
    "early_stopping_patience": 20,
    "gradient_clip_norm": 1.0,
    "auxiliary_loss_weight": 0.4,
    "dropout": 0.2,
    "drop_path": 0.1,
    "attention_dropout": 0.0,
    "seed": 41,
    "pretrained": False,
}
