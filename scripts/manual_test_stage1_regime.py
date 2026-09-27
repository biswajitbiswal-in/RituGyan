"""Manually test the saved Stage 1 model on a feature-store date."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.models.stage1_regime.classifier import (
    ML_REGIME_CODES,
    list_feature_store_dates,
    predict_regime_for_date,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Feature-store date in YYYY-MM-DD format")
    parser.add_argument("--list-dates", action="store_true", help="List all available dates and exit")
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
    args = parser.parse_args()

    if args.list_dates:
        print("Available feature-store dates:")
        for available_date in list_feature_store_dates(args.feature_store):
            print(available_date)
        return

    requested_date = args.date or input("Enter feature-store date (YYYY-MM-DD): ").strip()
    prediction = predict_regime_for_date(
        requested_date,
        artifact_dir=args.artifact_dir,
        store=args.feature_store,
    )

    print("RituGyan Manual Model Tester")
    print(f"\nDate: {prediction.date}")
    print(f"\nCandidate A label:\n{prediction.candidate_a_label}")
    print(f"\nML prediction:\n{prediction.predicted_regime}")
    print(f"\nConfidence:\n{prediction.confidence:.1%}")
    print("\nClass probabilities:")
    for regime in ML_REGIME_CODES:
        print(f"{regime} {prediction.probabilities[regime]:.1%}")
    print(f"\nResult:\n{'MATCH' if prediction.matches_candidate_a else 'MISMATCH'}")


if __name__ == "__main__":
    main()
