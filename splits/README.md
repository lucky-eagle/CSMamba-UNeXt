# Split manifests

These manifests were generated with `prepare_splits.py --seed 41` and use paths relative to the repository's `inputs/` directory.

- `clinicdb/`: video-sequence-grouped CVC-ClinicDB split, 490/61/61.
- `kvasir/`: image-level Kvasir-SEG split, 800/100/100.
- `combined/`: independent image-level split after pooling all 1,612 development images, 1,290/161/161.
- `external/`: complete CVC-300, CVC-ColonDB and ETIS evaluation sets.

`combined/test.csv` is retained to document the complete 8:1:1 split and is not a reported evaluation endpoint. Training copies all three relevant manifests into the run directory and records SHA-256 hashes in `config.resolved.json` and `checkpoint_metadata.json`.

The corrected manifests differ from earlier local artifacts that used 489/61/62 for CVC-ClinicDB and 1,289/161/162 for the pooled data. Results from those earlier artifacts require retraining before they can be attributed to the corrected protocol.
