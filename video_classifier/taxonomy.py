"""
Taxonomy of chunking-relevant video classes.

Each class is defined by *temporal structure and visual dynamics*, NOT semantic
content.  Two cooking videos (a talking-head recipe vs. a cinematic montage)
land in different classes because they require different chunking strategies.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Dict


class VideoClass(IntEnum):
    """Eight chunking-relevant video classes."""

    TALKING_HEAD = 0          # Podcasts, vlogs, lectures, interviews
    SCREEN_CAPTURE = 1        # Screen recordings, gameplay with HUD
    CINEMATIC_NARRATIVE = 2   # Movies, TV, music videos, scripted content
    CONTINUOUS_ACTION = 3     # Sports, egocentric, surveillance
    NATURE_OBSERVATIONAL = 4  # Wildlife, landscapes, slow nature docs
    MONTAGE_COMPILATION = 5   # Highlight reels, TikTok/Reels, rapid cuts
    PRESENTATION_SLIDES = 6   # Slideshows, webinars, text-heavy frames
    MULTI_PANEL = 7           # Split-screen, reaction vids, PiP, gallery


@dataclass(frozen=True)
class ChunkingStrategy:
    """Describes how a video class should be chunked."""

    name: str
    description: str
    default_params: Dict[str, float]


CHUNKING_STRATEGIES: Dict[VideoClass, ChunkingStrategy] = {
    VideoClass.TALKING_HEAD: ChunkingStrategy(
        name="transcript_aligned",
        description=(
            "Segment on speech pauses and topic shifts detected via NLP on "
            "the transcript.  Each chunk is a coherent spoken segment."
        ),
        default_params={"min_chunk_sec": 10.0, "max_chunk_sec": 120.0},
    ),
    VideoClass.SCREEN_CAPTURE: ChunkingStrategy(
        name="visual_diff",
        description=(
            "Segment when pixel-level difference between consecutive frames "
            "exceeds a threshold (app switch, scene transition)."
        ),
        default_params={"diff_threshold": 0.35, "min_chunk_sec": 5.0},
    ),
    VideoClass.CINEMATIC_NARRATIVE: ChunkingStrategy(
        name="shot_boundary",
        description=(
            "Detect hard cuts, dissolves, and wipes.  Group shots into "
            "scene-level chunks using visual similarity."
        ),
        default_params={"cut_threshold": 0.6, "scene_gap_sec": 30.0},
    ),
    VideoClass.CONTINUOUS_ACTION: ChunkingStrategy(
        name="temporal_window",
        description=(
            "Fixed or adaptive time-window with motion-energy gating.  "
            "Split long unbroken takes into digestible windows."
        ),
        default_params={"window_sec": 30.0, "motion_gate": 0.1},
    ),
    VideoClass.NATURE_OBSERVATIONAL: ChunkingStrategy(
        name="activity_burst",
        description=(
            "Segment on motion spikes in otherwise calm footage (animal "
            "appears, weather change, camera movement)."
        ),
        default_params={"burst_threshold": 0.25, "context_pad_sec": 5.0},
    ),
    VideoClass.MONTAGE_COMPILATION: ChunkingStrategy(
        name="pace_aware_shots",
        description=(
            "Every cut is a potential boundary; group rapid-fire shots by "
            "visual similarity to avoid single-frame chunks."
        ),
        default_params={"min_shot_group": 3, "similarity_threshold": 0.7},
    ),
    VideoClass.PRESENTATION_SLIDES: ChunkingStrategy(
        name="slide_transition",
        description=(
            "Detect slide changes via OCR-diff or template matching.  "
            "Each slide becomes one chunk."
        ),
        default_params={"text_diff_threshold": 0.3, "min_slide_sec": 3.0},
    ),
    VideoClass.MULTI_PANEL: ChunkingStrategy(
        name="panel_decompose",
        description=(
            "Detect panels/sub-windows, then chunk each sub-stream "
            "independently using the most appropriate strategy."
        ),
        default_params={"edge_strength": 0.5, "min_panel_area": 0.1},
    ),
}


# Convenience: class index ↔ label string
CLASS_NAMES = {c.value: c.name.lower() for c in VideoClass}
NAME_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}
