# fish_length

Consolidated fish-body-length pipeline: fetches `fish_measure_v3` API
responses for a folder of images, parses them into a shared `FishRecord`,
runs four independent 2D pixel-space measurement methods on the keypoints,
draws an overlay comparing all four against the API's own estimate, and
logs one CSV row per image. Built by dedup'ing two previously-unrefactored
notebook cells in `../nootebook.ipynb` (see "Provenance" below) — this
package is now the canonical home for that logic; the two source cells are
left in place, unmodified.

## File map

- `config.py` — shared constants: `STRAIGHT_THRESHOLD`, `BODY_INDICES`, `CM_PER_INCH`, `NAMED_KEYPOINT_MAP`.
- `record.py` — the `FishRecord` dataclass (see contract below).
- `geometry.py` — pure-math 2D primitives (`fit_line_pca`, `normalized_max_deviation`, `_project_extension`, `compute_oriented_bbox`). No I/O.
- `methods_2d.py` — `MeasurementResult` dataclass + the four 2D methods (`centroid`, `segment_sum`, `centroid_pivot`, `bbox`). Method definitions/quirks are documented in code comments there.
- `methods_3d.py` — **deferred stub, no implementation.** Docstring only. Do not implement unless explicitly asked.
- `fetch.py` — `fetch_folder()`: sequential (by design, unlike other threaded fetch tooling in this repo) POST-per-image against `fish_measure_v3`.
- `parsing.py` — `parse_api_records()`: turns `fetch_folder()`'s output JSON into `{filename: FishRecord}`.
- `report.py` — `draw_overlay()` + CSV logging + `run_2d(record, image_path, ...)`.

## FishRecord contract

```python
FishRecord(
    image_name, points, px_per_inch, estimated_length_in,
    depth=None, camera_intrinsic=None, camera_transform=None,
    estimated_length_3d_in=None, raw={},
)
```

`has_3d` is `True` only when `depth`, `camera_intrinsic`, and
`camera_transform` are all set. `parse_api_records` never sets them today —
`has_3d` is always `False` on records built from `fish_measure_v3`. It
exists so `methods_3d.py` can be filled in later (against a source that
actually returns depth/camera config) without changing this shape.

## Status — 2D implemented, 3D deferred (update this section when that changes)

- **2D**: fully implemented and round-tripped against real `fish_measure_v3`
  responses. `fetch.py` → `parsing.py` → `methods_2d.py` → `report.py` is
  the live path.
- **3D**: explicitly out of scope for now. `methods_3d.py` is a stub only.
  Don't implement it unless asked — the real 3D math lives in
  `../nootebook.ipynb`'s "Fish 3D length report" cell, past its corrupted
  splice (see Provenance).

## fish_measure_v3 field paths (confirmed against a real response)

Endpoint: `https://ai.flytechy.site/image_processing/fish_measure_v3`,
option `enableBumpboardReading` (bool, sent as multipart form data
alongside the image file).

This is a **different, newer schema** than the "detection JSON" format
Script A/B (see Provenance) were written against — field names below do
NOT match `owner.proofBalls` / `medias` / `fish3DMeasurements` from that
older format.

- Keypoints: `response["fish"][0]["keypoints"]` — a dict keyed by the same
  names as `config.NAMED_KEYPOINT_MAP` (`mouthLip0`, `mouthLip1`, `gills`,
  `body0`..`body4`, `tailStart`, `tailCorner0`, `tailCorner1`, `tailMidEnd`,
  `girthStart`, `girthEnd`), each value a `[x, y]` pixel pair.
- Calibration: `response["pballPixelDim"]` / `response["pballPhysicalDim"]`
  (pixels / inches → `px_per_inch`), directly at the response's top level.
  `response["bumpBoardLength"]` exists in the schema but was `null` in
  every sampled response — no bump-board calibration path is implemented.
- Estimated length: `response["estimatedLength"]`, already in inches (no
  `owner.preferredUnits` field exists on this endpoint to convert from).
- No image-URL field on this endpoint (the image was already local when
  POSTed) — `FishRecord.image_name` comes from `fetch_folder()`'s dict key.

## Conventions

- No hardcoded credentials or paths. `fetch.py` loads `USERNAME`/`PASSWORD`
  from `.env` via `python-dotenv`, and raises a clear error at call time if
  either is missing.
- `fetch.py` is sequential by design — do not add threading/retries here;
  that's a different pattern used elsewhere in this repo.
- One shared `MeasurementResult` (methods_2d.py) and one shared
  `FishRecord` (record.py) — don't reintroduce per-method or per-source
  return types.

## Provenance (for future sessions digging into this)

Source material was two un-refactored cells in `../nootebook.ipynb`:

- **Script A** = cell 8, "Fish 2D length report" — self-contained, not
  corrupted. The primary source for `methods_2d.py` / `geometry.py`.
- **Script B** = cell 6, "Fish 3D length report" — corrupted: its
  `NAMED_KEYPOINT_MAP = {` dict literal is cut off mid-definition and
  spliced directly into an entirely separate 2D batch-validation script
  (CVAT/COCO/CSV parsers, an old-vs-new-vs-bbox-vs-true-length comparison,
  `summary.csv` output — none of that is used here), before the real 3D
  content (`Point3D`, `convert_3d_to_world_coord`, `reproject_world_to_pixel`,
  `compute_length_centroid_3d`, `compute_length_centroid_pivot_3d`,
  `compute_bbox_length_3d`) resumes and finishes the file. Only the
  geometry primitives shared with Script A were pulled from Script B (and
  numerically cross-checked against Script A's copy); everything
  batch-validation- and 3D-specific was left in place, unrefactored.
