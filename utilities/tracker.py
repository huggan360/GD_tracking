"""The existing Ultralytics ByteTrack setup, kept separate from training."""
import math
from types import SimpleNamespace

from ultralytics.trackers.byte_tracker import BYTETracker


def new_tracker(sample_fps=5, max_gap_seconds=1.0):
    # Ultralytics treats track_buffer as a frame count at its default rate of 30.
    settings = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25, match_thresh=0.8, fuse_score=True, track_buffer=max(1, math.ceil(max_gap_seconds * sample_fps)))
    return BYTETracker(settings)
