"""Train and report the Stage 1 five-class weather-regime baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.models.stage1_regime.classifier import train_stage1_from_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-store",
        type=Path,
        default=repo_root / "data" / "processed" / "feature_store",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=repo_root / "models" / "regime_classifier" / "stage1_lightgbm_5class_candidate_a_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train_stage1_from_store(
        feature_store_dir=args.feature_store,
        artifact_dir=args.artifact_dir,
        random_seed=args.seed,
    )
    print(json.dumps({"artifact_dir": str(result.artifact_dir), "metrics": result.metrics}, indent=2))


if __name__ == "__main__":
    main()
