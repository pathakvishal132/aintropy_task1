"""
Feature extraction pipeline: video → 130-dimensional float vector.

Design principles:
  - Sample sparsely (16 uniform frames + 3 temporal bursts)
  - Extract cheap structural signals (histograms, edges, diffs)
  - Avoid per-frame deep-net embeddings at classification time
  - Total extraction time: 2–5 sec per video on CPU
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExtractionConfig:
    """Tuneable parameters for the feature extraction pipeline."""

    # Frame sampling
    n_uniform_frames: int = 16
    burst_positions: Tuple[float, ...] = (0.25, 0.50, 0.75)
    burst_fps: int = 5
    burst_duration_sec: float = 1.0

    # Image preprocessing
    resize_dim: int = 224  # downscale frames before feature computation

    # HSV histogram
    h_bins: int = 16
    s_bins: int = 16
    v_bins: int = 16

    # Temporal diff threshold for shot-transition counting
    shot_diff_threshold: float = 0.30

    # Face detection
    use_face_detection: bool = True
    face_min_confidence: float = 0.5

    # Text detection
    use_text_detection: bool = True

    # Audio features
    use_audio: bool = False  # disabled by default (requires ffmpeg extraction)


# ---------------------------------------------------------------------------
# Frame Sampling
# ---------------------------------------------------------------------------

def _sample_uniform_indices(total_frames: int, n: int) -> List[int]:
    """Return *n* frame indices spread uniformly across the video."""
    if total_frames <= n:
        return list(range(total_frames))
    step = total_frames / n
    return [int(step * i + step / 2) for i in range(n)]


def _sample_burst_indices(
    total_frames: int,
    fps: float,
    positions: Tuple[float, ...],
    burst_fps: int,
    burst_duration: float,
) -> List[List[int]]:
    """Return lists of frame indices for each temporal burst."""
    bursts: List[List[int]] = []
    frames_per_burst = int(burst_fps * burst_duration)
    for pos in positions:
        centre = int(total_frames * pos)
        half = int(fps * burst_duration / 2)
        start = max(0, centre - half)
        step = max(1, int(fps / burst_fps))
        indices = [start + i * step for i in range(frames_per_burst)]
        indices = [idx for idx in indices if idx < total_frames]
        bursts.append(indices)
    return bursts


def read_frames(
    video_path: str | Path,
    indices: List[int],
) -> List[np.ndarray]:
    """Read specific frame indices from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames: List[np.ndarray] = []
    sorted_indices = sorted(set(indices))
    idx_set = set(sorted_indices)

    for target in sorted_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            logger.warning("Failed to read frame %d from %s", target, video_path)

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Feature Group A: Color & Texture Statistics
# ---------------------------------------------------------------------------

def _hsv_histogram(frame: np.ndarray, cfg: ExtractionConfig) -> np.ndarray:
    """Compute a normalised HSV histogram for one frame."""
    small = cv2.resize(frame, (cfg.resize_dim, cfg.resize_dim))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [cfg.h_bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [cfg.s_bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [cfg.v_bins], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s, hist_v])
    total = hist.sum()
    return hist / total if total > 0 else hist


def _edge_density(frame: np.ndarray, cfg: ExtractionConfig) -> float:
    """Fraction of pixels that are edges (Canny)."""
    small = cv2.resize(frame, (cfg.resize_dim, cfg.resize_dim))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges)) / edges.size


def _colorfulness(frame: np.ndarray, cfg: ExtractionConfig) -> float:
    """Hasler & Süsstrunk colorfulness metric."""
    small = cv2.resize(frame, (cfg.resize_dim, cfg.resize_dim)).astype(np.float64)
    B, G, R = small[:, :, 0], small[:, :, 1], small[:, :, 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, mean_rg = float(rg.std()), float(rg.mean())
    std_yb, mean_yb = float(yb.std()), float(yb.mean())
    std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
    mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    return float(std_root + 0.3 * mean_root)


def _color_texture_features(
    frames: List[np.ndarray],
    cfg: ExtractionConfig,
) -> np.ndarray:
    """
    Per-frame: HSV histogram (h+s+v bins) + edge density + colorfulness.
    Aggregation: mean + std across frames.
    Returns: 2 × (h_bins + s_bins + v_bins + 2) dims.
    """
    per_frame_dim = cfg.h_bins + cfg.s_bins + cfg.v_bins + 2
    if not frames:
        return np.zeros(2 * per_frame_dim, dtype=np.float32)

    all_feats = []
    for f in frames:
        hist = _hsv_histogram(f, cfg)
        ed = _edge_density(f, cfg)
        cf = _colorfulness(f, cfg)
        all_feats.append(np.concatenate([hist, [ed, cf]]))

    stacked = np.stack(all_feats)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature Group B: Temporal Dynamics
# ---------------------------------------------------------------------------

def _temporal_features(
    bursts: List[List[np.ndarray]],
    cfg: ExtractionConfig,
) -> np.ndarray:
    """
    From each burst: frame-to-frame L1 diffs → (mean, std, max).
                     shot-transition count.
    Optical flow stats omitted for speed (add if GPU available).
    Returns: n_bursts × 4 dims = 12 dims (we pad/truncate to 18).
    """
    features: List[float] = []

    for burst_frames in bursts:
        if len(burst_frames) < 2:
            features.extend([0.0] * 6)
            continue

        diffs = []
        for i in range(1, len(burst_frames)):
            g1 = cv2.cvtColor(
                cv2.resize(burst_frames[i - 1], (cfg.resize_dim, cfg.resize_dim)),
                cv2.COLOR_BGR2GRAY,
            ).astype(np.float32) / 255.0
            g2 = cv2.cvtColor(
                cv2.resize(burst_frames[i], (cfg.resize_dim, cfg.resize_dim)),
                cv2.COLOR_BGR2GRAY,
            ).astype(np.float32) / 255.0
            diff = float(np.mean(np.abs(g1 - g2)))
            diffs.append(diff)

        diffs_arr = np.array(diffs)
        features.append(float(diffs_arr.mean()))
        features.append(float(diffs_arr.std()))
        features.append(float(diffs_arr.max()))
        # Shot-transition count
        shot_count = int((diffs_arr > cfg.shot_diff_threshold).sum())
        features.append(float(shot_count))
        # Placeholder for optical-flow mean, std (set to 0 for CPU-only)
        features.append(0.0)
        features.append(0.0)

    # Pad to exactly 18 dims
    while len(features) < 18:
        features.append(0.0)

    return np.array(features[:18], dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature Group C: Structural Layout Indicators
# ---------------------------------------------------------------------------

def _face_features(
    frames: List[np.ndarray],
    cfg: ExtractionConfig,
) -> Tuple[float, float]:
    """Mean face count and mean face-area ratio across sampled frames."""
    if not cfg.use_face_detection:
        return 0.0, 0.0

    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        )
    except Exception:
        logger.warning("Haar cascade not available; skipping face features")
        return 0.0, 0.0

    counts: List[int] = []
    area_ratios: List[float] = []

    for frame in frames:
        small = cv2.resize(frame, (cfg.resize_dim, cfg.resize_dim))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        counts.append(len(faces))
        if len(faces) > 0:
            total_area = sum(w * h for (_, _, w, h) in faces)
            area_ratios.append(total_area / (cfg.resize_dim ** 2))
        else:
            area_ratios.append(0.0)

    return float(np.mean(counts)), float(np.mean(area_ratios))


def _text_region_ratio(
    frames: List[np.ndarray],
    cfg: ExtractionConfig,
) -> float:
    """
    Estimate text-area ratio using morphological text detection (lightweight).
    Looks for horizontally-aligned high-contrast regions.
    """
    if not cfg.use_text_detection:
        return 0.0

    ratios: List[float] = []
    for frame in frames:
        small = cv2.resize(frame, (cfg.resize_dim, cfg.resize_dim))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # MSER-like approach: threshold + morphology
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_area = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            # Text-like regions: wider than tall, not too small
            if 2.0 < aspect < 20.0 and w > 20 and h > 5:
                text_area += w * h
        ratios.append(text_area / (cfg.resize_dim ** 2))

    return float(np.mean(ratios)) if ratios else 0.0


def _edge_symmetry(frame: np.ndarray, cfg: ExtractionConfig) -> float:
    """Left-right edge histogram correlation (detects symmetric layouts)."""
    small = cv2.resize(frame, (cfg.resize_dim, cfg.resize_dim))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    mid = cfg.resize_dim // 2
    left = edges[:, :mid].astype(np.float32).flatten()
    right = np.flip(edges[:, mid:], axis=1).astype(np.float32).flatten()
    if left.std() == 0 or right.std() == 0:
        return 0.0
    # Truncate to same length
    min_len = min(len(left), len(right))
    corr = float(np.corrcoef(left[:min_len], right[:min_len])[0, 1])
    return corr if not np.isnan(corr) else 0.0


def _structural_features(
    frames: List[np.ndarray],
    cfg: ExtractionConfig,
) -> np.ndarray:
    """
    Returns 7 dims: face_count_mean, face_area_mean, text_ratio,
                     edge_symmetry, dominant_regions (stub), aspect_ratio,
                     resolution_bucket.
    """
    face_count, face_area = _face_features(frames, cfg)
    text_ratio = _text_region_ratio(frames, cfg)

    # Edge symmetry on first frame (representative)
    sym = _edge_symmetry(frames[0], cfg) if frames else 0.0

    # Dominant region count (simplified: count connected components on quantised image)
    if frames:
        small = cv2.resize(frames[0], (cfg.resize_dim, cfg.resize_dim))
        quantized = (small // 64) * 64
        gray_q = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
        _, binary_q = cv2.threshold(gray_q, 128, 255, cv2.THRESH_BINARY)
        n_labels, _ = cv2.connectedComponents(binary_q)
        dominant_regions = float(n_labels)
    else:
        dominant_regions = 0.0

    # Aspect ratio and resolution from original frame size
    if frames:
        h, w = frames[0].shape[:2]
        aspect_ratio = w / max(h, 1)
        res = min(h, w)
        if res >= 2160:
            res_bucket = 4.0
        elif res >= 1080:
            res_bucket = 3.0
        elif res >= 720:
            res_bucket = 2.0
        else:
            res_bucket = 1.0
    else:
        aspect_ratio = 1.78  # default 16:9
        res_bucket = 2.0

    return np.array(
        [face_count, face_area, text_ratio, sym, dominant_regions, aspect_ratio, res_bucket],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Feature Group D: Audio Fingerprint (stub — requires ffmpeg audio decode)
# ---------------------------------------------------------------------------

def _audio_features(video_path: str | Path, cfg: ExtractionConfig) -> np.ndarray:
    """
    Returns 5 dims: speech_ratio, music_ratio, silence_ratio,
                     spectral_centroid_mean, spectral_centroid_std.
    Currently returns zeros; plug in Silero VAD / librosa when audio is available.
    """
    # TODO: implement with Silero VAD + librosa
    return np.zeros(5, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main Extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Extracts a fixed-size feature vector from a video file.

    Usage::

        extractor = FeatureExtractor()
        features = extractor.extract("/path/to/video.mp4")
        print(features.shape)  # (130,)
    """

    EXPECTED_DIM = 130  # Color/Texture 100 + Temporal 18 + Structural 7 + Audio 5

    def __init__(self, cfg: Optional[ExtractionConfig] = None) -> None:
        self.cfg = cfg or ExtractionConfig()

    def extract(self, video_path: str | Path) -> np.ndarray:
        """
        Extract features from *video_path*.

        Returns:
            np.ndarray of shape ``(130,)`` with dtype ``float32``.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        if total_frames < 1:
            logger.warning("Video %s has 0 frames; returning zeros", video_path)
            return np.zeros(self.EXPECTED_DIM, dtype=np.float32)

        # --- Sample frame indices ---
        uniform_indices = _sample_uniform_indices(total_frames, self.cfg.n_uniform_frames)
        burst_indices = _sample_burst_indices(
            total_frames,
            fps,
            self.cfg.burst_positions,
            self.cfg.burst_fps,
            self.cfg.burst_duration_sec,
        )

        # --- Read frames ---
        all_indices = list(set(uniform_indices + [i for b in burst_indices for i in b]))
        all_frames = read_frames(video_path, all_indices)

        # Build index → frame mapping
        sorted_unique = sorted(set(all_indices))
        idx_to_frame = {}
        for frame, idx in zip(all_frames, sorted_unique):
            idx_to_frame[idx] = frame

        uniform_frames = [idx_to_frame[i] for i in uniform_indices if i in idx_to_frame]
        burst_frame_lists = [
            [idx_to_frame[i] for i in b if i in idx_to_frame]
            for b in burst_indices
        ]

        # --- Extract feature groups ---
        color_tex = _color_texture_features(uniform_frames, self.cfg)   # 100
        temporal = _temporal_features(burst_frame_lists, self.cfg)       # 18
        structural = _structural_features(uniform_frames, self.cfg)     # 7
        audio = _audio_features(video_path, self.cfg)                   # 5

        feature_vec = np.concatenate([color_tex, temporal, structural, audio])

        # Sanity check
        if feature_vec.shape[0] != self.EXPECTED_DIM:
            logger.error(
                "Feature dimension mismatch: got %d, expected %d",
                feature_vec.shape[0],
                self.EXPECTED_DIM,
            )
            # Pad or truncate to expected dim
            padded = np.zeros(self.EXPECTED_DIM, dtype=np.float32)
            n = min(feature_vec.shape[0], self.EXPECTED_DIM)
            padded[:n] = feature_vec[:n]
            return padded

        return feature_vec
