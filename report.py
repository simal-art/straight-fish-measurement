"""Overlay drawing + CSV logging + run_2d(record), adapted from Script A's
generate_report_2d (nootebook.ipynb, "Fish 2D length report" cell).

Differences from Script A, both forced by FishRecord's shape:
  - No image download here. Script A downloaded the frame image from a
    detection-JSON media URL; fish_measure_v3 records have no such URL (the
    image was already local when fetch.py POSTed it), so run_2d takes
    image_path directly.
  - Overlay drawing for centroid/centroid_pivot uses the raw/proj points
    MeasurementResult.extra carries (see methods_2d.py) rather than the
    11-tuples Script A drew from directly -- the drawn lines are the same.
"""

import csv
import os

from PIL import Image, ImageDraw, ImageFont

from .methods_2d import bbox, centroid, centroid_pivot, segment_sum

FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

CSV_LOG_PATH = "fish_2d_measurements.csv"

CSV_FIELDNAMES = [
    'json_id', 'image_filename', 'fish_name', 'is_straight', 'dev',
    'method1_total_in', 'method1_fork_in',
    'method2_total_in',
    'method3_total_in', 'method3_fork_in',
    'method4_bbox_length_in', 'method4_bbox_width_in',
    'est_2d_in',
]


def _fmt_in(value_px, px_per_inch):
    if value_px is None or not px_per_inch:
        return ''
    return round(value_px / px_per_inch, 2)


def append_measurement_to_csv(csv_path, row):
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in CSV_FIELDNAMES})


def draw_overlay(image_path, points, results, px_per_inch, estimated_length_in, output_path):
    """results: {'centroid': MeasurementResult, 'segment_sum': ..., 'centroid_pivot': ..., 'bbox': ...}

    Draws (matching Script A's generate_report_2d):
        - keypoints (yellow dots)
        - centroid's path: head_point -> p9 -> tail_point (green), plus the
          raw-keypoint-to-fitted-line offset (thin red segments)
        - centroid_pivot's path: head_point -> centroid -> p9 -> tail_point (blue)
        - bbox's oriented rectangle (magenta)
        - all four totals and the record's own estimated_length_in (red)
    """
    r1, r2, r3, r4 = results['centroid'], results['segment_sum'], results['centroid_pivot'], results['bbox']

    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    font_size = max(20, img.width // 40)
    dot_r = max(6, img.width // 200)
    line_w = max(4, img.width // 250)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except OSError:
        font = ImageFont.load_default()

    for idx, (x, y) in points.items():
        draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r],
                     fill=(255, 220, 0), outline=(0, 0, 0), width=3)
        draw.text((x + dot_r + 3, y - dot_r - 3), str(idx), fill=(0, 0, 0), font=font,
                   stroke_width=2, stroke_fill=(255, 255, 255))

    if r1.is_straight and r1.head_point is not None:
        p9 = r1.extra['p9']
        draw.line([r1.head_point, p9, r1.tail_point], fill=(29, 158, 117), width=line_w)
        draw.line([r1.extra['head_raw'], r1.head_point], fill=(220, 30, 30), width=max(3, line_w // 2))
        draw.line([r1.extra['tail_raw'], r1.tail_point], fill=(220, 30, 30), width=max(3, line_w // 2))

    if r3.is_straight and r3.head_point is not None:
        centroid_pt, p9 = r3.extra['centroid'], r3.extra['p9']
        draw.line([r3.head_point, centroid_pt, p9, r3.tail_point], fill=(40, 110, 220), width=line_w)

    if r4.total_length is not None:
        corners = r4.extra['corners']
        n = len(corners)
        for i in range(n):
            draw.line([corners[i], corners[(i + 1) % n]], fill=(190, 0, 190), width=max(3, line_w // 2))

    black, red, blue, magenta = (0, 0, 0), (200, 0, 0), (20, 70, 180), (170, 0, 170)
    lines = []

    if r1.is_straight is None:
        lines.append(("Method 1: unavailable (missing keypoints)", black))
    elif not r1.is_straight:
        lines.append((f"Method 1: curved, no length  (dev={r1.dev:.4f})", black))
    elif r1.total_length is None:
        lines.append(("Method 1: straight, missing required points", black))
    else:
        total_in, fork_in = _fmt_in(r1.total_length, px_per_inch), _fmt_in(r1.fork_length, px_per_inch)
        suffix = f"{total_in} in" if total_in != '' else f"{r1.total_length:.1f} px"
        fork_suffix = f"{fork_in} in" if fork_in != '' else f"{r1.fork_length:.1f} px"
        lines.append((f"Method 1: {suffix}  (fork {fork_suffix})", black))

    if r2.total_length is not None:
        m2_in = _fmt_in(r2.total_length, px_per_inch)
        suffix = f"{m2_in} in" if m2_in != '' else f"{r2.total_length:.1f} px"
        lines.append((f"Method 2 (verified): {suffix}", black))

    if r3.is_straight and r3.total_length is not None:
        total_in, fork_in = _fmt_in(r3.total_length, px_per_inch), _fmt_in(r3.fork_length, px_per_inch)
        suffix = f"{total_in} in" if total_in != '' else f"{r3.total_length:.1f} px"
        fork_suffix = f"{fork_in} in" if fork_in != '' else f"{r3.fork_length:.1f} px"
        lines.append((f"Method 3 (Centroid): {suffix}  (fork {fork_suffix})", blue))

    if r4.total_length is not None:
        bbox_in = _fmt_in(r4.total_length, px_per_inch)
        suffix = f"{bbox_in} in" if bbox_in != '' else f"{r4.total_length:.1f} px"
        lines.append((f"Method 4 (BBox): {suffix}", magenta))

    if estimated_length_in is not None:
        lines.append((f"2D Est (from API): {estimated_length_in:.2f} in", red))
    if px_per_inch is None:
        lines.append(("(no calibration found -- showing pixel lengths where inches unavailable)", black))
    lines.append(("green = M1, blue = M3, magenta = M4 (BBox box)", black))

    y_cursor = img.height * 0.03
    for text, color in lines:
        draw.text((img.width * 0.03, y_cursor), text, fill=color, font=font,
                   stroke_width=max(2, font_size // 15), stroke_fill=(255, 255, 255))
        y_cursor += font_size * 1.3

    img.save(output_path, quality=92)
    return output_path


def run_2d(record, image_path, out_dir='.', output_path=None, csv_path=None):
    """Runs all four 2D methods on `record`, draws the overlay onto
    image_path, appends one row to csv_path (default CSV_LOG_PATH), and
    returns the output image path.
    """
    if csv_path is None:
        csv_path = CSV_LOG_PATH

    r1 = centroid(record.points)
    r2 = segment_sum(record.points)
    r3 = centroid_pivot(record.points)
    r4 = bbox(record.points)
    results = {'centroid': r1, 'segment_sum': r2, 'centroid_pivot': r3, 'bbox': r4}

    if output_path is None:
        output_path = os.path.join(out_dir, f"report2d_{record.image_name}")
    draw_overlay(image_path, record.points, results,
                 record.px_per_inch, record.estimated_length_in, output_path)

    suggested_fishes = record.raw.get('suggestedFishes') or []
    fish_name = suggested_fishes[0].get('commonName', '') if suggested_fishes else ''
    row = {
        'json_id': record.raw.get('id', ''),
        'image_filename': record.image_name,
        'fish_name': fish_name,
        'is_straight': r1.is_straight if r1.is_straight is not None else '',
        'dev': round(r1.dev, 5) if r1.dev is not None else '',
        'method1_total_in': _fmt_in(r1.total_length, record.px_per_inch),
        'method1_fork_in': _fmt_in(r1.fork_length, record.px_per_inch),
        'method2_total_in': _fmt_in(r2.total_length, record.px_per_inch),
        'method3_total_in': _fmt_in(r3.total_length, record.px_per_inch),
        'method3_fork_in': _fmt_in(r3.fork_length, record.px_per_inch),
        'method4_bbox_length_in': _fmt_in(r4.total_length, record.px_per_inch),
        'method4_bbox_width_in': _fmt_in(r4.extra.get('bbox_width'), record.px_per_inch),
        'est_2d_in': round(record.estimated_length_in, 2) if record.estimated_length_in is not None else '',
    }
    append_measurement_to_csv(csv_path, row)

    return output_path
