"""Train and evaluate the isolated GFS-only Stage 1 classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.features.operational_gfs import operational_feature_names  # noqa: E402
from src.models.stage1_regime.classifier import (  # noqa: E402
    CLASS_ID_TO_REGIME_CODE,
    WESTERN_DISTURBANCE_POLICY,
    load_stage1_datasets,
    train_stage1_model,
)


DEFAULT_STORE = repo_root / "data/processed/feature_store_operational_gfs"
DEFAULT_ARTIFACT = repo_root / "models/regime_classifier/stage1_lightgbm_5class_operational_gfs_v1"
BASELINE_ARTIFACT = repo_root / "models/regime_classifier/stage1_lightgbm_5class_candidate_a_v1"
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_split_metrics(metrics: Mapping[str, Any]) -> str:
    lines = []
    for split_name in ("train", "validation", "test"):
        split = metrics[split_name]
        lines.extend(
            [
                f"### {split_name.title()} ({split['sample_count']} samples)",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ]
        )
        lines.extend(f"| {metric} | {split[metric]:.6f} |" for metric in METRICS)
        lines.extend(["", "Confusion matrix (rows=true, columns=predicted):", "", "```text"])
        lines.extend(" ".join(str(value) for value in row) for row in split["confusion_matrix"])
        lines.extend(["```", "", "Per-class metrics:", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"])
        lines.extend(
            f"| {class_name} | {values['precision']:.6f} | {values['recall']:.6f} | "
            f"{values['f1']:.6f} | {values['support']} |"
            for class_name, values in split["per_class"].items()
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--baseline-artifact", type=Path, default=BASELINE_ARTIFACT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    store_config = _load_json(args.feature_store / "store_config.json")
    if store_config.get("feature_version") != "stage1_operational_gfs_v1":
        raise ValueError("Refusing to train: feature store is not stage1_operational_gfs_v1")
    provenance = store_config.get("provenance", {})
    if provenance.get("era5_used_for_predictors") is not False:
        raise ValueError("Refusing to train: operational store provenance does not exclude ERA5")
    if provenance.get("candidate_a_regenerated") is not False:
        raise ValueError("Refusing to train: operational store indicates regenerated labels")
    if args.artifact_dir.resolve() == args.baseline_artifact.resolve():
        raise ValueError("Operational model artifact path must not equal the retrospective baseline path")

    expected_names = operational_feature_names()
    saved_names = tuple(_load_json(args.feature_store / "feature_names.json"))
    if saved_names != expected_names:
        raise ValueError("Operational feature name list does not match the canonical operational order")

    partitions = load_stage1_datasets(args.feature_store)
    if any(part.feature_names != saved_names for part in partitions.values()):
        raise ValueError("Stage 1 feature transformation does not match operational feature names")
    result = train_stage1_model(
        partitions,
        artifact_dir=args.artifact_dir,
        random_seed=args.seed,
    )

    metadata_path = result.artifact_dir / "metadata.json"
    artifact_metadata = _load_json(metadata_path)
    artifact_metadata.update(
        {
            "model_version": "stage1_lightgbm_5class_operational_gfs_v1",
            "feature_version": store_config["feature_version"],
            "feature_store": str(args.feature_store.relative_to(repo_root)),
            "feature_provenance": provenance,
            "channel_source_map": store_config["channel_source_map"],
            "source_synoptic_features": "GFS prmsl, pwat, 10u, 10v only",
            "feature_names": "feature_names.json; exact 122-name list",
            "western_disturbance_policy": WESTERN_DISTURBANCE_POLICY,
            "comparison_baseline_artifact": str(args.baseline_artifact.relative_to(repo_root)),
            "training_seed": int(args.seed),
        }
    )
    metadata_path.write_text(json.dumps(artifact_metadata, indent=2), encoding="utf-8")

    baseline_metrics = _load_json(args.baseline_artifact / "evaluation_metrics.json")
    comparison = {
        "baseline": str(args.baseline_artifact.relative_to(repo_root)),
        "operational_model": str(args.artifact_dir.relative_to(repo_root)),
        "note": (
            "Same Candidate A proxy labels and chronological years, but separate predictor provenance. "
            "This is a descriptive comparison, not a controlled ablation or independent ground-truth evaluation."
        ),
        "splits": {},
    }
    for split_name in ("validation", "test"):
        comparison["splits"][split_name] = {
            "baseline": {metric: baseline_metrics[split_name][metric] for metric in METRICS},
            "operational_gfs": {metric: result.metrics[split_name][metric] for metric in METRICS},
            "operational_minus_baseline": {
                metric: result.metrics[split_name][metric] - baseline_metrics[split_name][metric]
                for metric in METRICS
            },
        }
    (result.artifact_dir / "baseline_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Operational GFS Stage 1 Evaluation",
        "",
        f"Model artifact: `{args.artifact_dir.relative_to(repo_root)}`",
        f"Feature store version: `{store_config['feature_version']}`",
        f"Feature count: {len(saved_names)}",
        f"Seed: {args.seed}; deterministic CPU training with one thread.",
        "",
        "The model uses GFS-only operational predictors and unchanged Candidate A proxy labels. "
        "Western Disturbance is excluded from ML partitions.",
        "",
        "## Split Metrics",
        "",
        _format_split_metrics(result.metrics),
        "## Retrospective Baseline Comparison",
        "",
        "| Split | Metric | Retrospective ERA5 baseline | Operational GFS | Difference |",
        "|---|---|---:|---:|---:|",
    ]
    for split_name in ("validation", "test"):
        for metric in METRICS:
            values = comparison["splits"][split_name]
            report_lines.append(
                f"| {split_name} | {metric} | {values['baseline'][metric]:.6f} | "
                f"{values['operational_gfs'][metric]:.6f} | "
                f"{values['operational_minus_baseline'][metric]:+.6f} |"
            )
    report_lines.extend(
        [
            "",
            "The comparison is descriptive: both models use the same ERA5-conditioned Candidate A labels; "
            "only predictor provenance changes. It does not establish independent meteorological truth.",
            "",
            "Detailed machine-readable metrics are in `evaluation_metrics.json`; provenance is in `metadata.json`.",
        ]
    )
    (result.artifact_dir / "evaluation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_dir": str(result.artifact_dir),
                "feature_count": len(saved_names),
                "sample_counts": {
                    name: int(part.features.shape[0]) for name, part in partitions.items()
                },
                "validation": result.metrics["validation"],
                "test": result.metrics["test"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
