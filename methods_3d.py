"""3D length measurement -- DEFERRED.

Script B (nootebook.ipynb, "Fish 3D length report" cell) has real, working
3D content past its corrupted splice: Point3D, convert_3d_to_world_coord,
reproject_world_to_pixel, compute_length_centroid_3d,
compute_length_centroid_pivot_3d, compute_bbox_length_3d. None of it is
implemented here yet -- this pass is 2D only (see fish_length/CLAUDE.md).

FishRecord already carries depth / camera_intrinsic / camera_transform and
has_3d for when this gets built, but parsing.py never populates them yet.
"""
