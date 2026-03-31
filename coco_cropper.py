from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image


@dataclass(frozen=True)
class CropBox:
    """Represents a crop window in absolute image coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


class CocoCropper:
    """Create per-object crops and per-crop COCO JSON files from a COCO dataset.

    This class reads a COCO JSON file, crops each object using its bbox with
    optional padding, writes the crop image, and writes a COCO JSON file that
    contains a single image and the corresponding annotation with segmentation
    coordinates transformed to the cropped image coordinate system.

    Objects with annotation area smaller than ``min_pixels_area`` are skipped
    and recorded in a CSV report saved next to the crop JSON output directory.
    """

    def __init__(
        self,
        image_dir: Path,
        json_file: Path,
        crop_images_output_dir: Path,
        crop_json_output_dir: Path,
        padding: int = 60,
        min_pixels_area: int = 700,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.json_file = Path(json_file)
        self.crop_images_output_dir = Path(crop_images_output_dir)
        self.crop_json_output_dir = Path(crop_json_output_dir)
        self.padding = int(padding)
        self.min_pixels_area = int(min_pixels_area)

    def run(self) -> None:
        """Process all annotations and generate crops and JSON files.

        Annotations with area smaller than ``min_pixels_area`` are skipped and
        recorded in a CSV file saved next to the crop JSON output directory.
        """
        coco = self._load_coco()

        images_by_id = {img["id"]: img for img in coco.get("images", [])}
        categories, category_id_remap = self._normalize_categories(coco.get("categories", []))

        self.crop_images_output_dir.mkdir(parents=True, exist_ok=True)
        self.crop_json_output_dir.mkdir(parents=True, exist_ok=True)

        ignored_rows: List[Dict[str, Any]] = []

        for ann in coco.get("annotations", []):
            image_info = images_by_id.get(ann.get("image_id"))
            if image_info is None:
                raise ValueError(f"Annotation {ann.get('id')} references missing image_id")

            area_value = ann.get("area")
            if area_value is None:
                area_value = 0

            if float(area_value) < self.min_pixels_area:
                ignored_rows.append(
                    {
                        "image": image_info.get("file_name"),
                        "object_id": ann.get("id"),
                        "area": area_value,
                    }
                )
                continue

            image_path = self.image_dir / image_info["file_name"]
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            with Image.open(image_path) as img:
                img_w, img_h = img.size
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    raise ValueError(f"Annotation {ann.get('id')} has invalid bbox")

                crop_box = self._compute_crop_box(bbox, img_w, img_h)
                crop = img.crop((crop_box.x0, crop_box.y0, crop_box.x1, crop_box.y1))

            output_image_name = self._build_output_name(
                image_info["file_name"], ann.get("id"), ann.get("category_id"), image_path.suffix
            )
            output_image_path = self.crop_images_output_dir / output_image_name
            crop.save(output_image_path)

            crop_json = self._build_crop_json(
                image_info=image_info,
                annotation=ann,
                categories=categories,
                crop_box=crop_box,
                output_image_name=output_image_name,
                original_image_size=(img_w, img_h),
                category_id_remap=category_id_remap,
            )

            output_json_path = self.crop_json_output_dir / (Path(output_image_name).stem + ".json")
            with output_json_path.open("w", encoding="utf-8") as f:
                json.dump(crop_json, f, ensure_ascii=False, indent=2)

        self._write_ignored_objects_csv(ignored_rows)

    def _load_coco(self) -> Dict[str, Any]:
        """Load the COCO JSON file."""
        with self.json_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_ignored_objects_csv(self, rows: List[Dict[str, Any]]) -> None:
        """Write the CSV report for ignored objects due to min area threshold."""
        import csv

        output_path = self.crop_json_output_dir / "ignored_objects_because_min_area.csv"
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "object_id", "area"])
            writer.writeheader()
            writer.writerows(rows)

    def _compute_crop_box(self, bbox: List[float], img_w: int, img_h: int) -> CropBox:
        """Compute a padded, clamped crop box from a COCO bbox."""
        x, y, w, h = bbox
        pad = self.padding

        x0 = max(0, math.floor(x - pad))
        y0 = max(0, math.floor(y - pad))
        x1 = min(img_w, math.ceil(x + w + pad))
        y1 = min(img_h, math.ceil(y + h + pad))

        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid crop box computed from bbox {bbox}")

        return CropBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def _build_output_name(self, file_name: str, ann_id: Any, category_id: Any, suffix: str) -> str:
        """Construct the output file name following the required convention."""
        base = Path(file_name).stem
        suffix = suffix or Path(file_name).suffix or ".jpg"
        # original, old. return f"{base}_obj_{ann_id}_class_{category_id}{suffix}"
        return f"{base}_obj_{ann_id}{suffix}"

    def _build_crop_json(
        self,
        image_info: Dict[str, Any],
        annotation: Dict[str, Any],
        categories: List[Dict[str, Any]],
        crop_box: CropBox,
        output_image_name: str,
        original_image_size: Tuple[int, int],
        category_id_remap: Dict[int, int],
    ) -> Dict[str, Any]:
        """Build a single-image COCO JSON for the crop."""
        img_w, img_h = original_image_size
        x, y, w, h = annotation["bbox"]

        crop_segmentation = self._transform_segmentation(annotation.get("segmentation"), crop_box)

        new_bbox = [x - crop_box.x0, y - crop_box.y0, w, h]

        image_entry = {
            "id": image_info["id"],
            "width": crop_box.width,
            "height": crop_box.height,
            "file_name": output_image_name,
            "distance_left_border": x + w,
            "distance_top_border": y + h,
        }

        annotation_entry = dict(annotation)
        annotation_entry["bbox"] = new_bbox
        annotation_entry["segmentation"] = crop_segmentation
        annotation_entry["image_id"] = image_info["id"]
        if "category_id" in annotation_entry:
            old_category_id = annotation_entry["category_id"]
            annotation_entry["category_id"] = category_id_remap.get(old_category_id, old_category_id)
        annotation_entry.setdefault("confidence", 1)

        return {
            "images": [image_entry],
            "annotations": [annotation_entry],
            "categories": categories,
        }

    def _normalize_categories(
        self, categories: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
        """Normalize category IDs for output JSON files."""
        remap: Dict[int, int] = {}
        normalized: List[Dict[str, Any]] = []
        for category in categories:
            if not isinstance(category, dict):
                normalized.append(category)
                continue
            name = category.get("name")
            cat_id = category.get("id")
            if name == "Unclassified" and cat_id == 1:
                updated = dict(category)
                updated["id"] = 4196
                remap[1] = 4196
                normalized.append(updated)
            else:
                normalized.append(category)
        return normalized, remap

    def _transform_segmentation(self, segmentation: Any, crop_box: CropBox) -> Any:
        """Transform segmentation coordinates into crop-relative coordinates.

        Supports polygon-style segmentation (list of lists). RLE segmentation is not
        transformed and will raise an error.
        """
        if segmentation is None:
            return segmentation

        if isinstance(segmentation, dict):
            raise ValueError("RLE segmentation is not supported for coordinate transformation")

        if not isinstance(segmentation, list):
            raise ValueError("Unsupported segmentation format")

        transformed: List[List[float]] = []
        for polygon in segmentation:
            if not isinstance(polygon, list):
                raise ValueError("Segmentation polygons must be lists")
            if len(polygon) % 2 != 0:
                raise ValueError("Segmentation polygon length must be even")

            new_poly: List[float] = []
            for i in range(0, len(polygon), 2):
                x = polygon[i]
                y = polygon[i + 1]
                new_poly.append(x - crop_box.x0)
                new_poly.append(y - crop_box.y0)
            transformed.append(new_poly)

        return transformed
