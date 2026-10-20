"""
Command-line interface for the video classifier.

Usage examples:

    # Train on synthetic data (cold-start bootstrap)
    python -m video_classifier train --output model.joblib

    # Classify a single video
    python -m video_classifier classify --model model.joblib --video input.mp4

    # Classify a directory of videos
    python -m video_classifier classify --model model.joblib --dir ./videos/

    # Extract features only (for debugging / analysis)
    python -m video_classifier features --video input.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .classifier import VideoClassifier
from .features import ExtractionConfig, FeatureExtractor
from .taxonomy import CLASS_NAMES, CHUNKING_STRATEGIES, VideoClass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("video_classifier")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    """Train classifier (on synthetic data if no dataset provided)."""
    clf = VideoClassifier()

    if args.dataset:
        data = np.load(args.dataset)
        X, y = data["X"], data["y"]
        logger.info("Loaded dataset: %d samples, %d features", X.shape[0], X.shape[1])
    else:
        logger.info(
            "No dataset provided — generating synthetic data (%d per class)",
            args.n_per_class,
        )
        X, y = VideoClassifier.generate_synthetic_data(n_per_class=args.n_per_class)

    clf.train(X, y, use_lightgbm=not args.no_lightgbm)

    # Quick self-evaluation
    preds = clf.model.predict(X)
    accuracy = float((preds == y).mean())
    logger.info("Training accuracy: %.2f%%", accuracy * 100)

    output = Path(args.output)
    clf.save(output)
    logger.info("Model saved → %s", output)


def cmd_classify(args: argparse.Namespace) -> None:
    """Classify one or more videos."""
    clf = VideoClassifier.from_pretrained(args.model)

    # Collect video paths
    video_paths: list[Path] = []
    if args.video:
        video_paths.append(Path(args.video))
    if args.dir:
        exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"}
        video_paths.extend(
            p for p in Path(args.dir).rglob("*") if p.suffix.lower() in exts
        )

    if not video_paths:
        logger.error("No video files found. Use --video or --dir.")
        sys.exit(1)

    results = []
    for vp in video_paths:
        logger.info("Classifying: %s", vp)
        pred = clf.classify(vp)
        result = {
            "video": str(vp),
            "class_id": pred.class_id,
            "class_name": pred.class_name,
            "confidence": round(pred.confidence, 4),
            "chunking_strategy": pred.chunking_strategy.name,
            "top_3": [
                {"class": name, "prob": round(prob, 4)}
                for _, name, prob in pred.top_k
            ],
        }
        results.append(result)
        print(json.dumps(result, indent=2))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        logger.info("Results written → %s", args.output_json)


def cmd_features(args: argparse.Namespace) -> None:
    """Extract and print the 130-dim feature vector for a video."""
    extractor = FeatureExtractor()
    features = extractor.extract(args.video)
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature vector (first 20): {features[:20]}")
    if args.output_npy:
        np.save(args.output_npy, features)
        logger.info("Features saved → %s", args.output_npy)


def cmd_info(args: argparse.Namespace) -> None:
    """Print the taxonomy and chunking strategies."""
    print("\n=== Video Classes & Chunking Strategies ===\n")
    for vc in VideoClass:
        strategy = CHUNKING_STRATEGIES[vc]
        print(f"  [{vc.value}] {vc.name}")
        print(f"      Strategy : {strategy.name}")
        print(f"      Desc     : {strategy.description}")
        print(f"      Defaults : {strategy.default_params}")
        print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video_classifier",
        description="Lightweight video classifier for chunking-aware video indexing.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Train the classifier")
    p_train.add_argument("--dataset", type=str, default=None, help="Path to .npz with X, y")
    p_train.add_argument("--n-per-class", type=int, default=500, help="Synthetic samples per class")
    p_train.add_argument("--output", type=str, default="model.joblib", help="Output model path")
    p_train.add_argument("--no-lightgbm", action="store_true", help="Use RandomForest instead")

    # --- classify ---
    p_clf = sub.add_parser("classify", help="Classify video(s)")
    p_clf.add_argument("--model", type=str, required=True, help="Path to trained model")
    p_clf.add_argument("--video", type=str, default=None, help="Path to a single video")
    p_clf.add_argument("--dir", type=str, default=None, help="Directory of videos")
    p_clf.add_argument("--output-json", type=str, default=None, help="Save results as JSON")

    # --- features ---
    p_feat = sub.add_parser("features", help="Extract features from a video")
    p_feat.add_argument("--video", type=str, required=True, help="Path to video")
    p_feat.add_argument("--output-npy", type=str, default=None, help="Save features as .npy")

    # --- info ---
    sub.add_parser("info", help="Print taxonomy and chunking strategies")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "train": cmd_train,
        "classify": cmd_classify,
        "features": cmd_features,
        "info": cmd_info,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
