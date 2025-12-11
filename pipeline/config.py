"""Central configuration for the dataset processing pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse


@dataclass(frozen=True)
class Paths:
    """Holds canonical project paths relative to the repo root."""

    project_root: Path

    @property
    def raw_dataset_dir(self) -> Path:
        return self.project_root / "data" / "raw" / "dataset"

    @property
    def processed_dataset_dir(self) -> Path:
        return self.project_root / "data" / "processed" / "dataset"

    @property
    def plantvillage_zip(self) -> Path:
        return self.raw_dataset_dir / "plantvillage.zip"

    @property
    def plantdoc_zip(self) -> Path:
        return self.raw_dataset_dir / "plantdoc.zip"

    @property
    def tomato_leaf_zip(self) -> Path:
        return self.raw_dataset_dir / "Tomato Leaf Dataset  A dataset for multiclass disease detection and classification.zip"

    @property
    def plantvillage_raw(self) -> Path:
        return self.processed_dataset_dir / "PlantVillage"

    @property
    def plantdoc_raw(self) -> Path:
        return self.processed_dataset_dir / "PlantDoc"

    @property
    def tomato_leaf_raw(self) -> Path:
        return self.processed_dataset_dir / "TomatoLeaf"

    @property
    def plantvillage_processed(self) -> Path:
        return self.processed_dataset_dir / "PlantVillage_processed"

    @property
    def plantdoc_processed(self) -> Path:
        return self.processed_dataset_dir / "PlantDoc_processed"

    @property
    def tomato_leaf_processed(self) -> Path:
        return self.processed_dataset_dir / "TomatoLeaf_processed"

    @property
    def combined_csv(self) -> Path:
        return self.processed_dataset_dir / "combined_dataset.csv"

    @property
    def log_file(self) -> Path:
        return self.project_root / "processing.log"


@dataclass(frozen=True)
class PipelineConfig:
    """Aggregates user-configurable knobs for the pipeline run."""

    paths: Paths
    datasets: tuple[str, ...] = ("plantvillage", "plantdoc", "tomatoleaf")
    overwrite_existing: bool = True
    log_level: str = "INFO"


def load_default_config(project_root: Path | None = None, datasets: tuple[str, ...] | None = None) -> PipelineConfig:
    """Create a default configuration for the repository."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    selected = datasets or ("plantvillage", "plantdoc", "tomatoleaf")
    return PipelineConfig(paths=Paths(project_root=project_root), datasets=selected)


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Process agricultural datasets")
    parser.add_argument("--datasets", nargs="*", default=["plantvillage", "plantdoc", "tomatoleaf"],
                        help="Datasets to process: plantvillage, plantdoc, tomatoleaf")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    cfg = load_default_config(datasets=tuple(args.datasets))
    return PipelineConfig(paths=cfg.paths, datasets=tuple(args.datasets), log_level=args.log_level)
