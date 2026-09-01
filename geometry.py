"""Pure 2D geometry primitives, shared by all four methods in methods_2d.py.

Verbatim from Script A (nootebook.ipynb, "Fish 2D length report" cell) --
cross-checked against Script B's real 3D content (nootebook.ipynb, "Fish 3D
length report" cell, the code that resumes after the corrupted splice) and
confirmed byte-for-byte identical for fit_line_pca, normalized_max_deviation
and _project_extension. compute_oriented_bbox has no 3D counterpart to
cross-check against (Script B's own copy of it only exists inside the
spliced-in 2D batch-validation script, which is out of scope for this pass
-- see fish_length/CLAUDE.md).

No I/O here, and 2D only -- 3D unprojection/reprojection is deferred to
methods_3d.py.
"""

import numpy as np

from .config import BODY_INDICES


def fit_line_pca(body_points):
    """Best-fit straight line through all body_points (total least squares
    via SVD) -- uses every point, not just the two endpoints."""
    centroid = body_points.mean(axis=0)
    _, _, vt = np.linalg.svd(body_points - centroid)
    return centroid, vt[0]


def normalized_max_deviation(body_points):
    """Fits the PCA line, then measures how far the worst point strays from
    it, normalized by the line's own span (scale-invariant)."""
    centroid, direction = fit_line_pca(body_points)
    projections = [np.dot(p - centroid, direction) for p in body_points]
    line_len = max(projections) - min(projections)
    deviations = [
        np.linalg.norm((p - centroid) - proj * direction)
        for p, proj in zip(body_points, projections)
    ]
    return (max(deviations) / line_len) if line_len else 0.0


def _project_extension(pts, candidate_idx, origin, direction, sign):
    """Perpendicular-projects each candidate onto the line through `origin`
    along `direction`, returns whichever extends furthest past `origin`."""
    best_len, best_proj, best_raw = -np.inf, None, None
    for idx in candidate_idx:
        if idx not in pts:
            continue
        p = pts[idx]
        s = np.dot(p - origin, direction) * sign
        if s > best_len:
            best_len, best_proj, best_raw = s, origin + sign * s * direction, p
    return best_len, best_proj, best_raw


def compute_oriented_bbox(points, orientation_indices=None, margin=0.0):
    """Bounding box rotated to align with the fish's body axis (PCA over
    orientation_indices, default BODY_INDICES).

    Returns (corners, bbox_length_px, bbox_width_px):
        corners        -- list of 4 (x, y) points, in order
        bbox_length_px -- extent ALONG the fish's body axis
        bbox_width_px  -- extent ACROSS the body axis
    Returns None if there aren't at least 2 orientation points and 1 total
    point available.
    """
    orientation_indices = orientation_indices or BODY_INDICES
    pts = {k: np.array(v, dtype=float) for k, v in points.items()}

    axis_pts = np.array([pts[i] for i in orientation_indices if i in pts])
    if len(axis_pts) < 2:
        return None

    all_pts = np.array(list(pts.values()))
    if len(all_pts) == 0:
        return None

    _, direction = fit_line_pca(axis_pts)
    angle = np.arctan2(direction[1], direction[0])

    def _rotate(xy, theta):
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([[c, -s], [s, c]])
        return xy @ r.T

    rotated = _rotate(all_pts, -angle)
    min_xy = rotated.min(axis=0) - margin
    max_xy = rotated.max(axis=0) + margin

    corners_rot = np.array([
        [min_xy[0], min_xy[1]],
        [max_xy[0], min_xy[1]],
        [max_xy[0], max_xy[1]],
        [min_xy[0], max_xy[1]],
    ])

    corners_img = _rotate(corners_rot, angle)
    bbox_length_px = max_xy[0] - min_xy[0]
    bbox_width_px = max_xy[1] - min_xy[1]
    return [tuple(c) for c in corners_img], bbox_length_px, bbox_width_px
