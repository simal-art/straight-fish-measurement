"""Shared support for nootebook.ipynb's older "detection JSON" schema
(fields at the record's top level, `owner.proofBalls`, `medias`) --
de-duplicates what used to be identical copy-pasted code across the
notebook's Section 1 (batch-validation pipeline), Section 3 (3D report),
and Section 4 (2D report / Script A).

- `_normalize_detection_records` / `parse_fish_detection_jsons`: parse that
  schema into keypoints + calibration. Shared by Section 3 and Section 4
  (Section 1 has its own, genuinely different, parsers for CVAT/COCO/JSONL).
- `compute_length_centroid` / `compute_length_centroid_pivot`: kept in
  their ORIGINAL 11-tuple-returning form -- unlike methods_2d.py's
  `centroid`/`centroid_pivot`, which repackage the same math into
  MeasurementResult. Shared by Section 1 and Section 4.

New code should use parsing.py / methods_2d.py instead -- this module is
for de-duplicating existing tuple-shaped call sites, not for writing new
ones against.
"""

import json
import os

import numpy as np

from .config import BODY_INDICES, CM_PER_INCH, NAMED_KEYPOINT_MAP, STRAIGHT_THRESHOLD
from .geometry import _project_extension, fit_line_pca, normalized_max_deviation


def _normalize_detection_records(json_paths):
    """Accepts a single dict, a single file path, a list of dicts/paths, or
    a directory (every "*.json" file inside it). Returns [(name, record), ...]
    where name is the basename of each record's IMAGE media URL.
    """
    if isinstance(json_paths, dict):
        json_paths = [json_paths]
    elif isinstance(json_paths, str):
        if os.path.isdir(json_paths):
            json_paths = [
                os.path.join(json_paths, fn)
                for fn in sorted(os.listdir(json_paths))
                if fn.lower().endswith('.json')
            ]
        else:
            json_paths = [json_paths]

    records = []
    for item in json_paths:
        if isinstance(item, dict):
            records.append(item)
        else:
            with open(item) as f:
                records.append(json.load(f))

    named = []
    for i, rec in enumerate(records):
        name = None
        for media in rec.get('medias', []):
            if media.get('fileType') == 'IMAGE' and media.get('url'):
                name = os.path.basename(media['url'])
                break
        if not name:
            name = f"{rec.get('id', f'unknown_{i}')}.png"
        named.append((name, rec))

    return named


def parse_fish_detection_jsons(json_paths, pball_physical_dim_in=None):
    """Returns (points_by_image, calibrations, estimated_lengths_in).

    Calibration priority: this record's own pballPixelDim/pballPhysicalDim
    (checked defensively -- absent/0 in every record seen so far) ->
    owner.proofBalls[0].size (a real registered ball diameter, when
    present) -> pball_physical_dim_in supplied by the caller -> fallback:
    fishPixelLength / estimatedLength (validated against a record with a
    real registered ball size: matched to 4 decimal places).
    """
    points_by_image = {}
    calibrations = {}
    estimated_lengths_in = {}

    for name, rec in _normalize_detection_records(json_paths):
        fish_list = rec.get('fish', [])
        if fish_list:
            named_kp = fish_list[0].get('keypoints', {})
            points = {
                NAMED_KEYPOINT_MAP[k]: (float(v[0]), float(v[1]))
                for k, v in named_kp.items() if k in NAMED_KEYPOINT_MAP
            }
            if points:
                points_by_image[name] = points

        pball_px = rec.get('pballPixelDim')
        owner_balls = rec.get('owner', {}).get('proofBalls') or []
        owner_ball_size = owner_balls[0].get('size') if owner_balls else None
        pball_phys = rec.get('pballPhysicalDim') or owner_ball_size or pball_physical_dim_in
        px_per_inch = None
        if pball_px and pball_phys:
            px_per_inch = pball_px / pball_phys
        else:
            fish_px_len = rec.get('fishPixelLength')
            raw_est_len = rec.get('estimatedLength')
            if fish_px_len and raw_est_len:
                px_per_inch = fish_px_len / raw_est_len
        if px_per_inch:
            calibrations[name] = {'px_per_inch': px_per_inch}

        est_len = rec.get('estimatedLength')
        if est_len is not None:
            unit = rec.get('owner', {}).get('preferredUnits', {}).get('length', 'INCH')
            if unit != 'INCH':
                est_len = est_len / CM_PER_INCH
            estimated_lengths_in[name] = float(est_len)

    return points_by_image, calibrations, estimated_lengths_in


def compute_length_centroid(points):
    """Needs points 3-9, plus 1, 2, 10, 11, and 12.

    Head side: line through point 9, PCA direction over BODY_INDICES (same
    direction the oriented bbox uses). Points 1, 2 projected onto it from
    point 9; farthest wins (head_ext).

    Tail side: a SEPARATE line through point 9, its own PCA fit over 8, 9,
    12, and midpoint(10, 11). Points 10, 11, AND 12 are all projected onto
    THIS line from point 9; farthest wins (tail_ext).

    total_length_px = head_ext + tail_ext. fork_length_px = head_ext alone
    ("nose to point 9").
    """
    if not all(i in points for i in BODY_INDICES):
        return None

    pts = {k: np.array(v, dtype=float) for k, v in points.items()}
    body = np.array([pts[i] for i in BODY_INDICES])
    all_points = [tuple(pts[i]) for i in sorted(pts.keys())]
    p3, p9 = pts[3], pts[9]

    dev = normalized_max_deviation(body)
    is_straight = dev < STRAIGHT_THRESHOLD

    if not is_straight:
        return None, False, None, body, None, None, None, None, all_points, dev, None

    if 12 not in pts or 1 not in pts or 2 not in pts or 10 not in pts or 11 not in pts:
        return None, True, None, body, None, None, None, None, all_points, dev, None

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

    fork_length_px = head_ext
    total_length = head_ext + tail_ext
    if fork_length_px > total_length:
        total_length = fork_length_px

    body_path = [tuple(head_proj), tuple(p9), tuple(tail_proj)]

    return total_length, True, body_path, body, head_raw, head_proj, tail_raw, tail_proj, all_points, dev, fork_length_px


def compute_length_centroid_pivot(points):
    """Needs points 3-9, plus 1, 2, and 12 (at least one of 10/11, not
    necessarily both).

    Head pivot: centroid of points 1, 2, 3 (not point 9). Body direction:
    centroid -> point 9. Points 10, 11, 12 projected onto that SAME line
    from point 9; farthest wins.

    fork_length_px = head_ext + (centroid-to-p9 distance) -- nose to point
    9, via the centroid. total_length_px = fork_length_px + tail_ext.
    body_path is [head_proj, centroid, p9, tail_proj].
    """
    if not all(i in points for i in BODY_INDICES):
        return None

    pts = {k: np.array(v, dtype=float) for k, v in points.items()}
    body = np.array([pts[i] for i in BODY_INDICES])
    all_points = [tuple(pts[i]) for i in sorted(pts.keys())]
    p3, p9 = pts[3], pts[9]

    dev = normalized_max_deviation(body)
    is_straight = dev < STRAIGHT_THRESHOLD

    if not is_straight:
        return None, False, None, body, None, None, None, None, all_points, dev, None

    if 12 not in pts or 1 not in pts or 2 not in pts:
        return None, True, None, body, None, None, None, None, all_points, dev, None

    p1, p2 = pts[1], pts[2]

    centroid = (p1 + p2 + p3) / 3.0
    body_dir = (p9 - centroid) / np.linalg.norm(p9 - centroid)
    mouth_dir = -body_dir

    t1 = np.dot(p1 - centroid, mouth_dir)
    intersection_1 = centroid + t1 * mouth_dir
    t2 = np.dot(p2 - centroid, mouth_dir)
    intersection_2 = centroid + t2 * mouth_dir

    if t1 >= t2:
        head_ext, head_proj, head_raw = t1, intersection_1, p1
    else:
        head_ext, head_proj, head_raw = t2, intersection_2, p2

    tail_ext, tail_proj, tail_raw = _project_extension(pts, [10, 11, 12], p9, body_dir, sign=+1)

    body_length = np.linalg.norm(p9 - centroid)
    fork_length_px = head_ext + body_length
    total_length = fork_length_px + tail_ext
    if fork_length_px > total_length:
        total_length = fork_length_px

    body_path = [tuple(head_proj), tuple(centroid), tuple(p9), tuple(tail_proj)]

    return total_length, True, body_path, body, head_raw, head_proj, tail_raw, tail_proj, all_points, dev, fork_length_px
