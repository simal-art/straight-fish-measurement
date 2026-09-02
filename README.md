# fish_length

Fish body-length measurement from `fish_measure_v3` API keypoints: fetches
API responses for a folder of images, measures each detected fish four
independent ways (2D pixel space), draws an overlay comparing them, and
logs one CSV row per fish.

3D measurement is not implemented yet (`methods_3d.py` is a stub).

## Setup

```bash
git clone git@github.com:simal-art/straight-fish-measurement.git
cd straight-fish-measurement
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in this directory with your API credentials:

```
USERNAME=your_username
PASSWORD=your_password

# Only needed if these ever change from the defaults.
FISHTECHY_BASE_URL=https://ai.flytechy.site
FISHTECHY_ENDPOINT_PATH=image_processing/fish_measure_v3
```

## Run

Run from the directory **containing** `fish_length/` (not from inside it —
the module's files import each other as a package):

```bash
python -m fish_length.run /path/to/your/images out_reports/
```

This will:
1. POST every image in that folder to the API, saving all responses to `out_reports/responses.json`.
2. Parse the responses into keypoint records.
3. Measure each detected fish four ways, draw an overlay, and log a row to `out_reports/fish_2d_measurements.csv`.

When it's done, `out_reports/` has:
- `responses.json` — raw API responses (keep it if you want to re-run measurement without re-calling the API)
- `report2d_<image>.jpg` — one annotated overlay per detected fish
- `fish_2d_measurements.csv` — one row per fish, with all four methods' lengths in inches

## Using it from your own script

```python
from fish_length.fetch import fetch_folder
from fish_length.parsing import parse_api_records
from fish_length.report import run_2d

fetch_folder("images/", "responses.json")
records = parse_api_records("responses.json")
for name, rec in records.items():
    run_2d(rec, f"images/{name}", out_dir="reports/")
```

For details on the module's internal structure and the four measurement
methods, see [CLAUDE.md](CLAUDE.md).
