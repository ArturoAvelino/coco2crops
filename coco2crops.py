from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

from coco_cropper import CocoCropper


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config file and normalize CLI-style keys."""
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


def _coerce_path_list(value: Any, field_name: str) -> List[Path]:
    """Normalize a config value into a list of Paths."""
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    if isinstance(value, Sequence):
        paths: List[Path] = []
        for item in value:
            if isinstance(item, (str, Path)):
                paths.append(Path(item))
            else:
                raise ValueError(f"{field_name} entries must be strings or paths")
        return paths
    raise ValueError(f"{field_name} must be a string/path or a list of strings/paths")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coco2crops",
        description="Generate per-object crops and COCO JSON files from a COCO dataset.",
    )

    parser.add_argument("--config", type=Path, help="Path to YAML configuration file")
    parser.add_argument(
        "--image_dir",
        type=Path,
        help="Directory containing input images (config supports a list of directories)",
    )
    parser.add_argument(
        "--json_file",
        type=Path,
        help="Path to COCO JSON file (config supports a list of JSON files)",
    )
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
    """Resolve CLI and config values into normalized settings."""
    config: Dict[str, Any] = {}
    if args.config:
        config = load_config(args.config)

    def pick(key: str, default: Any = None) -> Any:
        value = getattr(args, key)
        if value is not None:
            return value
        return config.get(key, default)

    settings_raw = {
        "image_dir": pick("image_dir"),
        "json_file": pick("json_file"),
        "crop_images_output_dir": pick("crop_images_output_dir"),
        "crop_json_output_dir": pick("crop_json_output_dir"),
        "padding": pick("padding", 60),
        "min_pixels_area": pick("min_pixels_area", 700),
    }

    missing = [k for k, v in settings_raw.items() if v is None]
    if missing:
        raise ValueError(f"Missing required settings: {', '.join(missing)}")

    image_dirs = _coerce_path_list(settings_raw["image_dir"], "image_dir")
    json_files = _coerce_path_list(settings_raw["json_file"], "json_file")
    if len(image_dirs) != len(json_files):
        raise ValueError("image_dir and json_file must have the same number of entries")

    return {
        "image_dirs": image_dirs,
        "json_files": json_files,
        "crop_images_output_dir": settings_raw["crop_images_output_dir"],
        "crop_json_output_dir": settings_raw["crop_json_output_dir"],
        "padding": settings_raw["padding"],
        "min_pixels_area": settings_raw["min_pixels_area"],
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    settings = resolve_settings(args)

    for image_dir, json_file in zip(settings["image_dirs"], settings["json_files"]):
        cropper = CocoCropper(
            image_dir=image_dir,
            json_file=json_file,
            crop_images_output_dir=Path(settings["crop_images_output_dir"]),
            crop_json_output_dir=Path(settings["crop_json_output_dir"]),
            padding=int(settings["padding"]),
            min_pixels_area=int(settings["min_pixels_area"]),
        )
        cropper.run()


if __name__ == "__main__":
    main()
