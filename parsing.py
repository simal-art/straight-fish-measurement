"""Parses fetch.py's combined JSON (fish_measure_v3 responses) into
FishRecords.

Real v3 field paths (confirmed against /home/simal/Downloads/image_responses.json,
a live fetch_folder() output covering 13 images) -- this is a DIFFERENT,
newer endpoint than the "detection JSON" schema Script A/B parse, and does
NOT share its field names:
    keypoints        response["fish"][0]["keypoints"]   -- a dict keyed by
                      the SAME names as config.NAMED_KEYPOINT_MAP
                      (mouthLip0, mouthLip1, gills, body0-4, tailStart,
                      tailCorner0/1, tailMidEnd, girthStart, girthEnd),
                      each value a [x, y] pixel pair.
    calibration       response["pballPixelDim"] / response["pballPhysicalDim"]
                      (pixels / inches -> px_per_inch), present directly at
                      the top level -- NOT nested under owner.proofBalls
                      the way the older detection-JSON schema has it.
                      response["bumpBoardLength"] exists in the schema but
                      was null in every sampled response (these test images
                      used a proof ball, not a bump board); no bump-board
                      calibration path is implemented here since no real
                      response with one populated has been seen yet.
    estimated length  response["estimatedLength"], already in inches -- no
                      owner.preferredUnits field exists on this endpoint to
                      convert from (unlike the older schema).
There is no "medias"/image-URL field on this endpoint's response (the image
was already local when it was POSTed) -- image_name comes from
fetch_folder()'s own dict key instead.
"""

import json

from .config import NAMED_KEYPOINT_MAP
from .record import FishRecord


def parse_api_records(records_json_path: str) -> dict[str, FishRecord]:
    """records_json_path: path to the JSON fetch_folder() produces
    ({filename: {"status_code", "response"}} or {"status_code": None,
    "error": ...} on a failed call).

    Skips (logs, doesn't raise on) any entry with status_code != 200, an
    "error" key, or no fish detected. 3D fields are left None.
    """
    with open(records_json_path) as f:
        records = json.load(f)

    out = {}
    for filename, entry in records.items():
        if "error" in entry or entry.get("status_code") != 200:
            print(f"skipping {filename}: status_code={entry.get('status_code')} "
                  f"error={entry.get('error')}")
            continue

        response = entry["response"]
        fish_list = response.get("fish") or []
        if not fish_list:
            print(f"skipping {filename}: no fish detected")
            continue

        named_kp = fish_list[0].get("keypoints", {})
        points = {
            NAMED_KEYPOINT_MAP[k]: (float(v[0]), float(v[1]))
            for k, v in named_kp.items() if k in NAMED_KEYPOINT_MAP
        }

        pball_px = response.get("pballPixelDim")
        pball_in = response.get("pballPhysicalDim")
        px_per_inch = (pball_px / pball_in) if (pball_px and pball_in) else None

        estimated_length_in = response.get("estimatedLength")

        out[filename] = FishRecord(
            image_name=filename,
            points=points,
            px_per_inch=px_per_inch,
            estimated_length_in=estimated_length_in,
            raw=response,
        )

    return out
