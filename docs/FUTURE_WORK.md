# Future Work

## Domain-Agnostic Support

- Generalize processors so they operate on declarative dataset schemas (folder layout, label rules) to support non-plant domains.
- Introduce plugin registry for new dataset modules (`pipeline/<domain>.py`) with discovery via entrypoints or config files.
- Provide metadata templates and validation to ensure compatibility with downstream ML pipelines regardless of source domain.
- Offer sample processors for other domains (e.g., materials, medical imaging) to demonstrate extensibility.

## Processed Dataset Loader

- Build a reusable dataset loader that can ingest any processed dataset (or the combined CSV) and expose helpers like `records()`, `iter_image_paths()`, and `to_dataframe()`.
- Include batching/shuffling utilities for direct integration with PyTorch/TensorFlow data pipelines.
- Surface CLI commands to export subsets or statistics from processed metadata, supporting ML experiment orchestration.
