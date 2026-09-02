"""Example end-to-end driver: fetch -> parse -> measure/report for a folder
of images. Run as a module from the directory CONTAINING fish_length/ (not
from inside it) since report.py/parsing.py import each other with package-
relative imports:

    python -m fish_length.run /path/to/images [out_dir]
"""

import os
import sys

from fish_length.fetch import fetch_folder
from fish_length.parsing import parse_api_records
from fish_length.report import run_2d


def main():
    image_folder = sys.argv[1] if len(sys.argv) > 1 else "images"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "reports"
    os.makedirs(out_dir, exist_ok=True)

    responses_json = os.path.join(out_dir, "responses.json")
    fetch_folder(image_folder, responses_json)

    records = parse_api_records(responses_json)
    print(f"parsed {len(records)} record(s) with a detected fish")

    for name, rec in records.items():
        image_path = os.path.join(image_folder, name)
        out_path = run_2d(rec, image_path, out_dir=out_dir)
        print(f"{name}: report saved to {out_path}")


if __name__ == "__main__":
    main()
