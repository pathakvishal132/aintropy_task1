"""
Lightweight video classifier using LightGBM (or scikit-learn fallback).

Supports:
  - Training on labelled feature vectors
  - Prediction on single vectors or batches
  - Synthetic-data bootstrapping for cold-start
  - Model persistence via joblib
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

from .features import ExtractionConfig, FeatureExtractor
from .taxonomy import CLASS_NAMES, CHUNKING_STRATEGIES, VideoClass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """Result of classifying a single video."""

    class_id: int
    class_name: str
    confidence: float
    top_k: List[Tuple[int, str, float]]  # [(class_id, class_name, prob), ...]

    @property
    def video_class(self) -> VideoClass:
        return VideoClass(self.class_id)

    @property
    def chunking_strategy(self):
        return CHUNKING_STRATEGIES[self.video_class]


class VideoClassifier:
    """
    End-to-end video → class prediction.

    Usage::

        clf = VideoClassifier.from_pretrained("model.joblib")
        pred = clf.classify("/path/to/video.mp4")
        print(pred.class_name, pred.confidence)
        print(pred.chunking_strategy)
    """

    def __init__(
        self,
        model=None,
        feature_extractor: Optional[FeatureExtractor] = None,
        confidence_threshold: float = 0.50,
    ) -> None:
        self.model = model
        self.extractor = feature_extractor or FeatureExtractor()
        self.confidence_threshold = confidence_threshold

    # ----- Training ----------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_lightgbm: bool = True,
    ) -> None:
        """
        Train the classifier on pre-extracted features.

        Args:
            X: Feature matrix of shape ``(n_samples, 130)``.
            y: Label array of shape ``(n_samples,)`` with values in ``[0, 7]``.
            use_lightgbm: If True, use LightGBM; otherwise RandomForest.
        """
        if use_lightgbm:
            try:
                import lightgbm as lgb

                self.model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    num_leaves=31,
                    min_child_samples=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                    verbose=-1,
                    objective="multiclass",
                    num_class=len(VideoClass),
                )
                self.model.fit(X, y)
                logger.info("Trained LightGBM classifier on %d samples", len(y))
                return
            except ImportError:
                logger.warning("LightGBM not available; falling back to RandomForest")

        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X, y)
        logger.info("Trained RandomForest classifier on %d samples", len(y))

    # ----- Prediction --------------------------------------------------------

    def predict_features(self, features: np.ndarray) -> Prediction:
        """Classify a pre-extracted 130-dim feature vector."""
        if self.model is None:
            raise RuntimeError("No model loaded — call train() or from_pretrained()")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        probs = self.model.predict_proba(features)[0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        top_k = sorted(
            [(i, CLASS_NAMES[i], float(probs[i])) for i in range(len(probs))],
            key=lambda t: t[2],
            reverse=True,
        )[:3]

        return Prediction(
            class_id=class_id,
            class_name=CLASS_NAMES[class_id],
            confidence=confidence,
            top_k=top_k,
        )

    def classify(self, video_path: str | Path) -> Prediction:
        """Full pipeline: extract features → classify."""
        features = self.extractor.extract(video_path)
        pred = self.predict_features(features)

        if pred.confidence < self.confidence_threshold:
            logger.warning(
                "Low confidence %.2f for %s — defaulting to cinematic_narrative",
                pred.confidence,
                video_path,
            )
            # Fall back to the most general strategy
            fallback_id = VideoClass.CINEMATIC_NARRATIVE.value
            pred = Prediction(
                class_id=fallback_id,
                class_name=CLASS_NAMES[fallback_id],
                confidence=pred.confidence,
                top_k=pred.top_k,
            )

        return pred

    def classify_batch(self, video_paths: List[str | Path]) -> List[Prediction]:
        """Classify multiple videos. Embarrassingly parallel — use with joblib."""
        return [self.classify(p) for p in video_paths]

    # ----- Persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize model to disk."""
        joblib.dump(self.model, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        cfg: Optional[ExtractionConfig] = None,
    ) -> "VideoClassifier":
        """Load a previously trained model."""
        model = joblib.load(model_path)
        return cls(model=model, feature_extractor=FeatureExtractor(cfg))

    # ----- Synthetic data for bootstrapping ----------------------------------

    @staticmethod
    def generate_synthetic_data(
        n_per_class: int = 500,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data with class-specific feature distributions.

        This encodes domain priors about each video class:
          - talking_head:      high face count, high speech, low motion
          - screen_capture:    high text ratio, low colorfulness, medium motion
          - cinematic_narrative: high colorfulness, moderate shot transitions
          - continuous_action: high motion, low shot transitions
          - nature_observational: low motion, high colorfulness, no faces
          - montage_compilation: very high shot transitions, high motion
          - presentation_slides: very high text, very low motion
          - multi_panel:       high edge symmetry, multiple faces

        Returns:
            (X, y) where X is (n_per_class * 8, 130) and y is (n_per_class * 8,).
        """
        rng = np.random.RandomState(seed)
        n_classes = len(VideoClass)
        dim = FeatureExtractor.EXPECTED_DIM

        X_all, y_all = [], []

        # Indices into the 130-dim vector (approximations for synthetic data):
        #   [0:100]   color/texture (mean+std of HSV hist + edge + colorful)
        #   [100:118] temporal dynamics
        #   [118:125] structural layout
        #   [125:130] audio

        for class_id in range(n_classes):
            base = rng.randn(n_per_class, dim).astype(np.float32) * 0.1

            if class_id == VideoClass.TALKING_HEAD:
                base[:, 118] = rng.uniform(1, 3, n_per_class)     # face count
                base[:, 119] = rng.uniform(0.05, 0.3, n_per_class) # face area
                base[:, 125] = rng.uniform(0.6, 1.0, n_per_class)  # speech ratio
                base[:, 100:106] = rng.uniform(0, 0.05, (n_per_class, 6))  # low motion

            elif class_id == VideoClass.SCREEN_CAPTURE:
                base[:, 120] = rng.uniform(0.1, 0.6, n_per_class)  # text ratio
                # Low colorfulness in mean (around index 49 in color/texture)
                base[:, 49] = rng.uniform(0, 30, n_per_class)
                base[:, 100:106] = rng.uniform(0.05, 0.15, (n_per_class, 6))

            elif class_id == VideoClass.CINEMATIC_NARRATIVE:
                base[:, 49] = rng.uniform(40, 100, n_per_class)    # high colorfulness
                base[:, 103] = rng.uniform(1, 5, n_per_class)      # shot transitions
                base[:, 109] = rng.uniform(1, 5, n_per_class)

            elif class_id == VideoClass.CONTINUOUS_ACTION:
                base[:, 100:106] = rng.uniform(0.15, 0.4, (n_per_class, 6))  # high motion
                base[:, 103] = rng.uniform(0, 1, n_per_class)      # few shots
                base[:, 109] = rng.uniform(0, 1, n_per_class)

            elif class_id == VideoClass.NATURE_OBSERVATIONAL:
                base[:, 100:106] = rng.uniform(0, 0.03, (n_per_class, 6))  # very low motion
                base[:, 49] = rng.uniform(50, 120, n_per_class)    # high colorfulness
                base[:, 118] = 0                                    # no faces

            elif class_id == VideoClass.MONTAGE_COMPILATION:
                base[:, 103] = rng.uniform(5, 15, n_per_class)     # many shots
                base[:, 109] = rng.uniform(5, 15, n_per_class)
                base[:, 115] = rng.uniform(5, 15, n_per_class)
                base[:, 100:103] = rng.uniform(0.1, 0.3, (n_per_class, 3))

            elif class_id == VideoClass.PRESENTATION_SLIDES:
                base[:, 120] = rng.uniform(0.3, 0.8, n_per_class)  # very high text
                base[:, 100:106] = rng.uniform(0, 0.02, (n_per_class, 6))  # nearly static
                base[:, 49] = rng.uniform(0, 20, n_per_class)      # low colorfulness

            elif class_id == VideoClass.MULTI_PANEL:
                base[:, 121] = rng.uniform(0.3, 0.8, n_per_class)  # high edge symmetry
                base[:, 118] = rng.uniform(2, 6, n_per_class)      # multiple faces
                base[:, 119] = rng.uniform(0.02, 0.15, n_per_class)

            X_all.append(base)
            y_all.append(np.full(n_per_class, class_id, dtype=np.int32))

        return np.vstack(X_all), np.concatenate(y_all)
