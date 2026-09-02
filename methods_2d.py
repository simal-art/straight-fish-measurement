"""The four 2D length-measurement methods, from Script A (nootebook.ipynb,
"Fish 2D length report" cell), renamed for clarity:

    Script A name                  -> here
    compute_length_centroid        -> centroid
    compute_method_2_segment_sum   -> segment_sum
    compute_length_centroid_pivot  -> centroid_pivot
    compute_oriented_bbox (wrapped)-> bbox

Geometric logic, thresholds and math are unchanged from Script A. What
changed is the return type: Script A's bare 11-tuples (or a lone float, or
None) are replaced by one shared MeasurementResult, so report.py can call
all four methods uniformly.

MeasurementResult's six named fields are the numeric contract every method
fills in the same way. Not every method has a head/tail/fork/straightness
concept the way centroid and centroid_pivot do (segment_sum never gates on
straightness in Script A; bbox has no fork or straightness concept at all)
-- those fields are simply left None where Script A never computed them.
`extra` is a judgment-call addition (not in the original spec) to carry
method-specific rendering geometry (e.g. bbox's 4 corners, the raw
pre-projection keypoints) that report.py needs to reproduce Script A's
overlay, without forcing every method's odd shape into the six core fields.
"""

from dataclasses import dataclass, field

import numpy as np

from .config import BODY_INDICES, STRAIGHT_THRESHOLD
from .geometry import _project_extension, compute_oriented_bbox, fit_line_pca, normalized_max_deviation


@dataclass
class MeasurementResult:
    total_length: float | None
    fork_length: float | None
    is_straight: bool | None
    dev: float | None
    head_point: tuple | None
    tail_point: tuple | None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# METHOD 1 -- centroid (point-9-anchored, the "primary" method).
#
# Quirk preserved from Script A: fork_length is head_ext ALONE, i.e. "nose
# to point 9" -- NOT "nose to the tail notch" the way the 3D script's
# Method 1 defines fork length. This is deliberate, already-tested 2D
# behavior; don't "fix" it to match the 3D definition.
#
# Straightness gate: every method that has one uses normalized_max_deviation
# over BODY_INDICES (points 3-9) against STRAIGHT_THRESHOLD, computed once
# and shared as `dev`/`is_straight` here. segment_sum and bbox don't apply
# this gate in Script A (see their docstrings below).
# ---------------------------------------------------------------------------
def centroid(points):
    """Needs points 3-9, plus 1, 2, 10, 11, and 12.

    Head side: line through point 9, PCA direction over BODY_INDICES (same
    direction the oriented bbox uses). Points 1, 2 projected onto it from
    point 9; farthest wins (head_ext).

    Tail side: a SEPARATE line through point 9, its own PCA fit over 8, 9,
    12, and midpoint(10, 11). Points 10, 11, AND 12 are all projected onto
    THIS line from point 9; farthest wins (tail_ext).

    total_length = head_ext + tail_ext. fork_length = head_ext alone.
    """
    if not all(i in points for i in BODY_INDICES):
        return MeasurementResult(None, None, None, None, None, None)

    pts = {k: np.array(v, dtype=float) for k, v in points.items()}
    body = np.array([pts[i] for i in BODY_INDICES])
    p3, p9 = pts[3], pts[9]

    dev = normalized_max_deviation(body)
    is_straight = dev < STRAIGHT_THRESHOLD
    if not is_straight:
        return MeasurementResult(None, None, False, dev, None, None)

    if 12 not in pts or 1 not in pts or 2 not in pts or 10 not in pts or 11 not in pts:
        return MeasurementResult(None, None, True, dev, None, None)

    p10, p11, p12 = pts[10], pts[11], pts[12]

    _, head_dir = fit_line_pca(body)
    if np.dot(p3 - p9, head_dir) < 0:
        head_dir = -head_dir
    head_ext, head_proj, head_raw = _project_extension(pts, [1, 2], p9, head_dir, sign=+1)

    p8 = pts[8]
    mid_10_11 = (p10 + p11) / 2.0
    tail_ref_pts = np.array([p8, p9, p12, mid_10_11])
    _, tail_dir = fit_line_pca(tail_ref_pts)
    if np.dot(p12 - p9, tail_dir) < 0:
        tail_dir = -tail_dir
    tail_ext, tail_proj, tail_raw = _project_extension(pts, [10, 11, 12], p9, tail_dir, sign=+1)

    fork_length = head_ext
    total_length = head_ext + tail_ext
    if fork_length > total_length:
        total_length = fork_length

    return MeasurementResult(
        total_length, fork_length, True, dev,
        tuple(head_proj), tuple(tail_proj),
        extra={'p9': tuple(p9), 'head_raw': tuple(head_raw), 'tail_raw': tuple(tail_raw)},
    )


# ---------------------------------------------------------------------------
# METHOD 2 -- segment_sum: projected consecutive segments (mouth -> 3 -> 4
# -> ... -> 9), a cross-check that should always equal Method 1's total
# exactly (telescoping sum along the same line). Uses the SAME tail_ext
# definition as centroid (10, 11, 12 all candidates).
#
# Script A never applies the straightness gate here, and never draws an
# overlay path for it (only a text line) -- preserved as-is: is_straight
# and dev are always None, and there's no head/tail point to return.
# ---------------------------------------------------------------------------
def segment_sum(points):
    if not all(i in points for i in BODY_INDICES):
        return MeasurementResult(None, None, None, None, None, None)
    required = list(BODY_INDICES) + [1, 2, 10, 11, 12]
    if not all(i in points for i in required):
        return MeasurementResult(None, None, None, None, None, None)

    pts = {k: np.array(v, dtype=float) for k, v in points.items()}
    p3, p9 = pts[3], pts[9]
    body = np.array([pts[i] for i in BODY_INDICES])

    _, head_dir = fit_line_pca(body)
    if np.dot(p3 - p9, head_dir) < 0:
        head_dir = -head_dir

    mouth_idx = 1 if np.dot(pts[1] - p9, head_dir) > np.dot(pts[2] - p9, head_dir) else 2
    chain_idx = [mouth_idx, 3, 4, 5, 6, 7, 8, 9]
    proj = {i: p9 + np.dot(pts[i] - p9, head_dir) * head_dir for i in chain_idx}
    head_side_total = sum(
        np.linalg.norm(proj[chain_idx[i + 1]] - proj[chain_idx[i]])
        for i in range(len(chain_idx) - 1)
    )

    p8, p10, p11, p12 = pts[8], pts[10], pts[11], pts[12]
    mid_10_11 = (p10 + p11) / 2.0
    tail_ref_pts = np.array([p8, p9, p12, mid_10_11])
    _, tail_dir = fit_line_pca(tail_ref_pts)
    if np.dot(p12 - p9, tail_dir) < 0:
        tail_dir = -tail_dir
    tail_ext, _, _ = _project_extension(pts, [10, 11, 12], p9, tail_dir, sign=+1)

    return MeasurementResult(head_side_total + tail_ext, None, None, None, None, None)


# ---------------------------------------------------------------------------
# METHOD 3 -- centroid_pivot.
# ---------------------------------------------------------------------------
def centroid_pivot(points):
    """Needs points 3-9, plus 1, 2, and 12 (at least one of 10/11, not
    necessarily both).

    Head pivot: centroid of points 1, 2, 3 (not point 9). Body direction:
    centroid -> point 9. Points 10, 11, 12 projected onto that SAME line
    from point 9; farthest wins.

    fork_length = head_ext + (centroid-to-p9 distance) -- nose to point 9,
    via the centroid. total_length = fork_length + tail_ext.
    """
    if not all(i in points for i in BODY_INDICES):
        return MeasurementResult(None, None, None, None, None, None)

    pts = {k: np.array(v, dtype=float) for k, v in points.items()}
    body = np.array([pts[i] for i in BODY_INDICES])
    p3, p9 = pts[3], pts[9]

    dev = normalized_max_deviation(body)
    is_straight = dev < STRAIGHT_THRESHOLD
    if not is_straight:
        return MeasurementResult(None, None, False, dev, None, None)

    if 12 not in pts or 1 not in pts or 2 not in pts:
        return MeasurementResult(None, None, True, dev, None, None)

    p1, p2 = pts[1], pts[2]

    pivot = (p1 + p2 + p3) / 3.0
    body_dir = (p9 - pivot) / np.linalg.norm(p9 - pivot)
    mouth_dir = -body_dir

    t1 = np.dot(p1 - pivot, mouth_dir)
    intersection_1 = pivot + t1 * mouth_dir
    t2 = np.dot(p2 - pivot, mouth_dir)
    intersection_2 = pivot + t2 * mouth_dir

    if t1 >= t2:
        head_ext, head_proj, head_raw = t1, intersection_1, p1
    else:
        head_ext, head_proj, head_raw = t2, intersection_2, p2

    tail_ext, tail_proj, tail_raw = _project_extension(pts, [10, 11, 12], p9, body_dir, sign=+1)

    body_length = np.linalg.norm(p9 - pivot)
    fork_length = head_ext + body_length
    total_length = fork_length + tail_ext
    if fork_length > total_length:
        total_length = fork_length

    return MeasurementResult(
        total_length, fork_length, True, dev,
        tuple(head_proj), tuple(tail_proj),
        extra={'centroid': tuple(pivot), 'p9': tuple(p9),
               'head_raw': tuple(head_raw), 'tail_raw': tuple(tail_raw)},
    )


# ---------------------------------------------------------------------------
# METHOD 4 -- bbox: oriented bounding box, aligned to the fish's body axis.
#
# No fork or straightness concept in Script A (compute_oriented_bbox is
# called unconditionally, not gated on straightness) and no natural
# head/tail pair -- those fields stay None. bbox_width_px (needed for
# report.py's CSV column) and the 4 corners (needed to draw the rectangle)
# don't fit the six core fields, so they live in `extra`.
# ---------------------------------------------------------------------------
def bbox(points):
    result = compute_oriented_bbox(points)
    if result is None:
        return MeasurementResult(None, None, None, None, None, None)

    corners, bbox_length_px, bbox_width_px = result
    return MeasurementResult(
        bbox_length_px, None, None, None, None, None,
        extra={'corners': corners, 'bbox_width': bbox_width_px},
    )
