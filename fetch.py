"""Calls the fish_measure_v3 endpoint over a folder of images and dumps
every response into one JSON. Sequential by design (unlike other threaded
fetch tooling elsewhere in the repo) -- one image at a time, no retries.
"""

import json
import os

import requests
from dotenv import load_dotenv

DEFAULT_URL = "https://ai.flytechy.site/image_processing/fish_measure_v3"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def fetch_folder(image_folder: str, output_json: str,
                  url: str = DEFAULT_URL,
                  enable_bumpboard_reading: bool = True) -> dict:
    """POSTs every image in image_folder to `url`, one at a time, and writes
    the combined {filename: {"status_code", "response"}} dict to
    output_json. Returns the same dict.

    Requires USERNAME and PASSWORD in the environment (loaded from a local
    .env via python-dotenv) for HTTP basic auth.
    """
    load_dotenv()
    username = os.environ.get("USERNAME")
    password = os.environ.get("PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "USERNAME and PASSWORD must be set (in the environment or a .env "
            "file) to call fish_measure_v3")

    results = {}
    for filename in sorted(os.listdir(image_folder)):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue
        image_path = os.path.join(image_folder, filename)
        print(f"Processing: {filename}")
        try:
            with open(image_path, "rb") as image_file:
                response = requests.post(
                    url, auth=(username, password),
                    files={"file": (filename, image_file, "image/jpeg")},
                    data={"enableBumpboardReading": str(enable_bumpboard_reading).lower()},
                )
            try:
                response_data = response.json()
            except ValueError:
                response_data = response.text
            results[filename] = {"status_code": response.status_code, "response": response_data}
            print(f"  Status: {response.status_code}")
        except Exception as e:
            results[filename] = {"status_code": None, "error": str(e)}
            print(f"  ERROR: {e}")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Call fish_measure_v3 over a folder of images.")
    ap.add_argument("image_folder", help="folder of images to process")
    ap.add_argument("output_json", help="path to write the combined JSON results to")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--no-bumpboard", action="store_false", dest="enable_bumpboard_reading")
    args = ap.parse_args()

    fetch_folder(args.image_folder, args.output_json,
                 url=args.url, enable_bumpboard_reading=args.enable_bumpboard_reading)
