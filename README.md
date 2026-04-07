# coco2crops

Generate per-object cropped images and per-crop COCO JSON files from a COCO-format dataset. Each crop is created from the object `bbox` with optional padding, and each output JSON contains a single image and the corresponding annotation with segmentation coordinates transformed into the cropped image coordinate system.

## Features

- Crops every object listed in `annotations` using its `bbox` plus padding.
- Skips objects with annotation area below `min_pixels_area` and records them in a CSV report.
- Transforms polygon segmentation coordinates into crop-relative coordinates.
- Writes a COCO JSON file per crop containing one `images` entry and one `annotations` entry.
- Adds `distance_left_border` and `distance_top_border` to the output `images` entry.
- Generates output file names as `<original>_obj_<id>.<ext>`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create a YAML file (example: `config_files/config_example.yaml`):

```yaml
image_dir: /path/to/images
json_file: /path/to/segmentation.json
crop_images_output_dir: /path/to/output/crop/images
crop_json_output_dir: /path/to/output/crop/json
padding: 60
min_pixels_area: 700
```

You can also provide multiple image directories and COCO JSON files by using lists
of equal length (example: `config_files/config_example_2.yaml`):

```yaml
image_dir:
  - /path/to/images/sample_1/
  - /path/to/images/sample_2/
json_file:
  - /path/to/sample_1.json
  - /path/to/sample_2.json
crop_images_output_dir: /path/to/output/crop/images
crop_json_output_dir: /path/to/output/crop/json
padding: 60
min_pixels_area: 700
```

Each `image_dir` entry is paired with the `json_file` at the same index.

## Usage

```bash
./coco2crops --config config_example.yaml
```

You can also override config values with CLI flags:

```bash
./coco2crops \
  --config config_example.yaml \
  --padding 80
```

Or run directly via Python:

```bash
python3 coco2crops.py --config config_example.yaml
```

## Output Details

For each annotation:

- Crop window is computed as `bbox` plus padding, clamped to the image bounds.
- Output image size is `crop_width` × `crop_height`.
- Output image file name is:
  - `<original_stem>_obj_<annotation_id>.<original_ext>`
- Output JSON file name matches the output image name with a `.json` suffix.
- Annotations with `area < min_pixels_area` are ignored and recorded in a CSV report.

The output JSON file includes:

- `images[0].file_name`: cropped image file name
- `images[0].width` / `images[0].height`: crop dimensions
- `images[0].distance_left_border`: `bbox_right = x + width`
- `images[0].distance_top_border`: `bbox_bottom = y + height`
- `annotations[0].segmentation`: crop-relative polygon coordinates
- `annotations[0].bbox`: crop-relative bbox coordinates

The ignored-objects report is written to:

- `<crop_json_output_dir>/ignored_objects_because_min_area.csv`
- Columns: `image`, `object_id`, `area`

## Notes

- Only polygon segmentations (list-of-lists) are supported. If an annotation has RLE segmentation (`iscrowd = 1`), the script raises an error.
- `bbox` values are treated as floats but crop bounds are clamped to integer pixel coordinates. Segmentation coordinates remain floats.
- `distance_left_border` and `distance_top_border` are measured relative to the original image's top-left origin.

## Example

If the original image is `HM13-E_r5c4.jpg` and the annotation has `id = 319`, the outputs are:

- `HM13-E_r5c4_obj_319.jpg`
- `HM13-E_r5c4_obj_319.json`
