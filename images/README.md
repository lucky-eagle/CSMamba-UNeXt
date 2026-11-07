# Figure assets

This directory contains figures used by the project documentation. For the
verified environment, data layout, fixed splits, training commands,
evaluation metrics and ablation variants, see the repository-level
[`README.md`](../README.md).

The manuscript protocol uses 256x256 inputs, seed 41, AdamW with a learning
rate of `5e-4`, at most 200 epochs, validation-mIoU checkpoint selection and
early stopping after 20 consecutive epochs without improvement.
