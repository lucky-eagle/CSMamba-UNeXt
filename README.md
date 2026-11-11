# CSMamba-UNeXt

Official code for **CSMamba-UNeXt**, a state-space-enhanced encoder-decoder for colorectal polyp segmentation.

![CSMamba-UNeXt architecture](images/Fig. 1.tif)

## Environment

The manuscript experiments used Windows, Python 3.13.3, PyTorch 2.11.0, CUDA 12.8 and one NVIDIA GeForce RTX 5070 Ti. The package versions recorded from that environment are listed in `requirements-windows.txt`.

```powershell
conda env create -f environment.yml
conda activate csmamba-unext
```

`mamba-ssm` may require a CUDA-compatible local build on Windows. A successful import and a one-batch smoke test must be completed before starting a reported experiment.

## Data layout

The included manifests use the following relative paths. A mask may be placed directly under `masks/` or under `masks/0/` when regenerating manifests.

```text
inputs/
├── CVC-ClinicDB/
│   ├── images/
│   └── masks/
├── Kvasir/
│   ├── images/
│   └── masks/
└── polyp/TestDataset/
    ├── CVC-300/{images,masks}/
    ├── CVC-ColonDB/{images,masks}/
    └── ETIS-LaribPolypDB/{images,masks}/
```

The five public datasets are available from their official project pages:

- CVC-ClinicDB: <https://polyp.grand-challenge.org/CVCClinicDB/>
- Kvasir-SEG: <https://datasets.simula.no/kvasir-seg/>
- CVC-300 and CVC-ColonDB: <https://pages.cvc.uab.es/CVC-Colon/>
- ETIS-LaribPolypDB: <https://polyp.grand-challenge.org/ETISLarib/>

## Fixed experimental protocols

The repository uses CSV manifests so that every result has an explicit relationship to its training data, validation-selected checkpoint and test data. The bundled manifests were generated with seed 41 and contain:

| Protocol | Train | Validation | Test or reserved test | Partition rule |
|---|---:|---:|---:|---|
| `clinicdb` | 490 | 61 | 61 | CVC-ClinicDB, grouped by video sequence |
| `kvasir` | 800 | 100 | 100 | Kvasir-SEG, image-level random split |
| `combined` | 1,290 | 161 | 161 | all 1,612 development images pooled, then independently split at image level |

The `combined/test.csv` subset is retained only to make the 8:1:1 procedure explicit. It is not evaluated or reported. The best pooled-data checkpoint is fixed after validation and is then evaluated without retraining or fine-tuning on CVC-300, CVC-ColonDB and ETIS.

Regenerate and validate the manifests after placing the datasets under `inputs/`:

```powershell
python prepare_splits.py --data-root inputs --output splits --seed 41
```

`prepare_splits.py` performs an independent pooled split; it does not concatenate the two learning-experiment splits. It also fails if the required 490/61/61 grouped split or the expected dataset totals cannot be produced. The CVC-ClinicDB frame-to-sequence mapping used by the bundled split is recorded in `metadata/cvc_clinicdb_sequence_map.csv`.

## Training

Each command trains a separate instance of the same architecture and creates its own validation-selected checkpoint.

```powershell
python train.py --protocol clinicdb --run-name learning_clinicdb
python train.py --protocol kvasir --run-name learning_kvasir
python train.py --protocol combined --run-name generalization_pooled
```

The validation-selected defaults for CSMamba-UNeXt are: 256x256 input, random initialization, batch size 4, at most 200 epochs, AdamW with learning rate `5e-4`, betas `(0.9, 0.999)` and weight decay `1e-4`, cosine annealing to `1e-5`, no warm-up, FP16 AMP, gradient-norm clipping at 1.0, and early stopping after 20 consecutive epochs without validation mIoU improvement. The loss is `BCEWithLogits + soft Dice`; the auxiliary loss has weight 0.4. The manuscript Methods section must use `5e-4` when reporting results produced by the optimized checkpoints.

## Validation-only learning-rate tuning

The initial `1e-3` setting and the `5e-4` candidate used identical manifests, seed, architecture, loss, augmentation and checkpoint-selection rule. The learning rate was selected from validation mIoU only; no internal or external test result was used to select it. With `5e-4`, the best validation mIoU changed from 0.7640 to 0.7974 on CVC-ClinicDB, from 0.8322 to 0.8562 on Kvasir-SEG, and from 0.8211 to 0.8324 on the pooled protocol. Final test results and all metric-level deltas are stored in `experiments/optimized_lr5e4_20261022/OPTIMIZATION_REPORT.md`.

For every run, the output directory contains:

```text
experiments/<run-name>/
├── best_miou.pt
├── last.pt
├── checkpoint_metadata.json
├── config.resolved.json
├── training_log.csv
└── manifests/{train,val,test}.csv
```

The resolved configuration and checkpoint metadata record the split-manifest hashes. Resuming is rejected when those hashes differ.

## Evaluation

Learning capability uses each dataset-specific checkpoint only on its corresponding fixed test subset:

```powershell
python val.py `
  --checkpoint experiments/learning_clinicdb/best_miou.pt `
  --target CVC-ClinicDB=splits/clinicdb/test.csv `
  --output-dir experiments/learning_clinicdb/evaluation

python val.py `
  --checkpoint experiments/learning_kvasir/best_miou.pt `
  --target Kvasir-SEG=splits/kvasir/test.csv `
  --output-dir experiments/learning_kvasir/evaluation
```

Generalization loads the pooled-data checkpoint once and evaluates all three external datasets in the same process:

```powershell
python val.py `
  --checkpoint experiments/generalization_pooled/best_miou.pt `
  --target CVC-300=splits/external/cvc-300.csv `
  --target CVC-ColonDB=splits/external/cvc-colondb.csv `
  --target ETIS=splits/external/etis.csv `
  --output-dir experiments/generalization_pooled/external_evaluation
```

The evaluator writes one summary JSON, per-image IoU/Dice/B-F1/MAE CSV files, and the same checkpoint SHA-256 for all three external datasets. It refuses to evaluate the pooled reserved subset through the reported generalization route.

The seven reported metrics are computed with one shared implementation: image-wise mIoU and mDice at a 0.5 threshold, Boundary-F1 (B-F1), weighted F-measure, structure measure, maximum enhanced-alignment measure across 256 thresholds, and MAE from the continuous probability map. For B-F1, one-pixel contours are extracted by 3x3 erosion and matched within `ceil(0.008 * image diagonal)` pixels, which equals three pixels at 256x256. All dataset scores are image-wise macro averages.

The complete model's five manuscript result rows, including B-F1 and the checkpoint SHA-256 used for each row, are provided in `results/manuscript_results.csv`.

## Preprocessing and augmentation

RGB images are resized with bilinear interpolation; masks use nearest-neighbour interpolation and are binarized at 128. RGB intensities are scaled to `[0,1]` and normalized with mean and standard deviation `[0.5,0.5,0.5]`, mapping them approximately to `[-1,1]`.

During training, horizontal and vertical flips are applied independently with probability 0.5. With probability 0.5, a rotation is sampled uniformly from 90, 180 and 270 degrees. Validation and test inputs receive no stochastic augmentation.

## Ablation and baseline experiments

Every ablation implementation reuses the same three manifest sets: two dataset-specific training runs for learning capability and one pooled-data run for generalization. The `--model-name` argument exposes the seven variants reported in the manuscript:

| Variant | `--model-name` | Included components |
|---:|---|---|
| V1 | `Baseline-UNeXt` | UNeXt backbone |
| V2 | `Baseline+CSMamba` | backbone + CSMamba Block |
| V3 | `Baseline+ShallowEnhance+SEA+CA-Ghost` | complete model without CSMamba Block |
| V4 | `Baseline+CSMamba+SEA+CA-Ghost` | complete model without ShallowEnhance |
| V5 | `Baseline+CSMamba+ShallowEnhance+CA-Ghost` | complete model without SEA Attention |
| V6 | `Baseline+CSMamba+ShallowEnhance+SEA` | complete model without CA-Ghost Block |
| V7 | `CSMamba-UNeXt` | complete model |

For example, the V2 learning-capability run on CVC-ClinicDB is started with:

```powershell
python train.py --protocol clinicdb --model-name "Baseline+CSMamba" --run-name v2_clinicdb
```

Use the same `--model-name` for the Kvasir-SEG and pooled-data runs of that variant. Each checkpoint stores its model name, resolved configuration and manifest hashes, so evaluation reconstructs the matching architecture automatically.

The nine comparison methods are maintained in their own public codebases. Their architecture-specific preprocessing, initialization and training settings should be retained, while the manifests, validation-mIoU checkpoint rule, 20-epoch early stopping rule and evaluator are shared. No value should be copied from an original publication.

## Important compatibility note

Older scripts in the submitted archive used a 90/10 split, inconsistent checkpoint names, stochastic test-time rotation, batch-pooled metrics and double image normalization. They also contained an invalid Python class identifier. Those paths have been replaced. Checkpoints trained with the old double-normalization or different split manifests are not compatible evidence for the corrected protocol and must not be reported as if they were generated by this code.

For the complete architecture, `params.py` counts 11,956,815 trainable parameters (11.957 M). In the verified Windows environment, THOP reports 3.624 GMACs for a 256x256 input, corresponding to approximately 7.248 GFLOPs when one multiply-accumulate is counted as two floating-point operations. These are the values reported in the manuscript. Profiler support for selective-scan custom operations can affect FLOP totals, so comparisons must use the same environment and counting convention.
