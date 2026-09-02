"""Calls the fish_measure_v3 endpoint over a folder of images and dumps
every response into one JSON. Sequential by design (unlike other threaded
fetch tooling elsewhere in the repo) -- one image at a time, no retries.
"""

import json
import os

import requests
from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://ai.flytechy.site"
DEFAULT_ENDPOINT_PATH = "image_processing/fish_measure_v3"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# Resolved relative to this file, not the caller's cwd -- load_dotenv()'s
# default upward-search from cwd never finds fish_length/.env when this
# module is invoked from elsewhere (e.g. `python -m fish_length.run` from
# the parent directory), and can silently pick up an unrelated .env instead.
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _resolve_url(url=None, base_url=None, endpoint_path=None):
    """`url` (a full URL) wins if given. Otherwise builds base_url +
    endpoint_path, each falling back to FISHTECHY_BASE_URL /
    FISHTECHY_ENDPOINT_PATH in the environment, then the hardcoded default.
    """
    if url:
        return url
    base_url = base_url or os.environ.get("FISHTECHY_BASE_URL", DEFAULT_BASE_URL)
    endpoint_path = endpoint_path or os.environ.get("FISHTECHY_ENDPOINT_PATH", DEFAULT_ENDPOINT_PATH)
    return f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"


def fetch_folder(image_folder: str, output_json: str,
                  url: str = None, base_url: str = None, endpoint_path: str = None,
                  enable_bumpboard_reading: bool = True) -> dict:
    """POSTs every image in image_folder to the resolved endpoint, one at a
    time, and writes the combined {filename: {"status_code", "response"}}
    dict to output_json. Returns the same dict.

    Endpoint resolution (see _resolve_url): pass `url` for a full override,
    or `base_url`/`endpoint_path` to override just one piece -- otherwise
    falls back to FISHTECHY_BASE_URL / FISHTECHY_ENDPOINT_PATH in the
    environment (loaded from a local .env via python-dotenv), then the
    hardcoded defaults.

    Requires USERNAME and PASSWORD in the environment (loaded the same way)
    for HTTP basic auth.
    """
    # override=True: USERNAME in particular can collide with an ambient shell var
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    username = os.environ.get("USERNAME")
    password = os.environ.get("PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "USERNAME and PASSWORD must be set (in the environment or a .env "
            "file) to call fish_measure_v3")

    url = _resolve_url(url, base_url, endpoint_path)

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
    ap.add_argument("--url", default=None, help="full endpoint URL override")
    ap.add_argument("--base-url", default=None,
                    help="overrides FISHTECHY_BASE_URL / the default base URL")
    ap.add_argument("--endpoint-path", default=None,
                    help="overrides FISHTECHY_ENDPOINT_PATH / the default endpoint path")
    ap.add_argument("--no-bumpboard", action="store_false", dest="enable_bumpboard_reading")
    args = ap.parse_args()

    fetch_folder(args.image_folder, args.output_json,
                 url=args.url, base_url=args.base_url, endpoint_path=args.endpoint_path,
                 enable_bumpboard_reading=args.enable_bumpboard_reading)
