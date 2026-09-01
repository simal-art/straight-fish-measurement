"""Shared constants for the 2D fish-length pipeline.

Values are verbatim from Script A (nootebook.ipynb, the "Fish 2D length
report" cell) and cross-checked against the real 3D content in Script B
(nootebook.ipynb, the "Fish 3D length report" cell, past its corrupted
splice) -- both copies are identical.
"""

# Max normalized PCA deviation (see geometry.normalized_max_deviation) for a
# fish to be classified "straight" and have a length computed at all.
STRAIGHT_THRESHOLD = 0.02

# Core body chain (gills .. tailStart) used for the straightness gate and as
# the PCA axis for head-side projections and the oriented bbox.
BODY_INDICES = [3, 4, 5, 6, 7, 8, 9]

CM_PER_INCH = 2.54

# Maps a detection JSON's named keypoints onto the numeric 1-14 scheme used
# throughout this pipeline.
NAMED_KEYPOINT_MAP = {
    'mouthLip0': 1, 'mouthLip1': 2, 'gills': 3,
    'body0': 4, 'body1': 5, 'body2': 6, 'body3': 7, 'body4': 8,
    'tailStart': 9, 'tailCorner0': 10, 'tailCorner1': 11,
    'tailMidEnd': 12, 'girthStart': 13, 'girthEnd': 14,
}
