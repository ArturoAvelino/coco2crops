from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from coco_cropper import CocoCropper


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping")
    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(key, str) and key.startswith("--"):
            normalized[key[2:]] = value
        else:
            normalized[key] = value
    return normalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coco2crops",
        description="Generate per-object crops and COCO JSON files from a COCO dataset.",
    )

    parser.add_argument("--config", type=Path, help="Path to YAML configuration file")
    parser.add_argument("--image_dir", type=Path, help="Directory containing input images")
    parser.add_argument("--json_file", type=Path, help="Path to COCO JSON file")
    parser.add_argument("--crop_images_output_dir", type=Path, help="Output directory for cropped images")
    parser.add_argument("--crop_json_output_dir", type=Path, help="Output directory for crop JSON files")
    parser.add_argument("--padding", type=int, default=None, help="Padding (pixels) added around bbox")
    parser.add_argument(
        "--min_pixels_area",
        type=int,
        default=None,
        help="Minimum annotation area in pixels required to process an object",
    )

    return parser


def resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if args.config:
        config = load_config(args.config)

    def pick(key: str, default: Any = None) -> Any:
        value = getattr(args, key)
        if value is not None:
            return value
        return config.get(key, default)

    settings = {
        "image_dir": pick("image_dir"),
        "json_file": pick("json_file"),
        "crop_images_output_dir": pick("crop_images_output_dir"),
        "crop_json_output_dir": pick("crop_json_output_dir"),
        "padding": pick("padding", 60),
        "min_pixels_area": pick("min_pixels_area", 700),
    }

    missing = [k for k, v in settings.items() if v is None]
    if missing:
        raise ValueError(f"Missing required settings: {', '.join(missing)}")

    return settings


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    settings = resolve_settings(args)

    cropper = CocoCropper(
        image_dir=Path(settings["image_dir"]),
        json_file=Path(settings["json_file"]),
        crop_images_output_dir=Path(settings["crop_images_output_dir"]),
        crop_json_output_dir=Path(settings["crop_json_output_dir"]),
        padding=int(settings["padding"]),
        min_pixels_area=int(settings["min_pixels_area"]),
    )
    cropper.run()


if __name__ == "__main__":
    main()
