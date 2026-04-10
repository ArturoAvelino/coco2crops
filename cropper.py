from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import os

from tqdm import tqdm
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from queue import Queue
from threading import Thread
import cv2

def _compute_crop_box(bbox: List[float], img_w: int, img_h: int, padding=40) -> CropBox:
    """Compute a padded, clamped crop box from a COCO bbox."""
    x, y, w, h = bbox

    x0 = max(0, math.floor(x - padding))
    y0 = max(0, math.floor(y - padding))
    x1 = min(img_w, math.ceil(x + w + padding))
    y1 = min(img_h, math.ceil(y + h + padding))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop box computed from bbox {bbox}")

    return CropBox(x0=x0, y0=y0, x1=x1, y1=y1)

def _build_output_name(file_name: str, ann_id: Any, category_id: Any, suffix: str) -> str:
    """Construct the output file name following the required convention."""
    base = Path(file_name).stem
    suffix = suffix or Path(file_name).suffix or ".jpg"
    # original, old. return f"{base}_obj_{ann_id}_class_{category_id}{suffix}"
    return f"{base}_obj_{ann_id}{suffix}"

def _build_crop_json(
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

    crop_segmentation = _transform_segmentation(annotation.get("segmentation"), crop_box)

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

def _transform_segmentation(segmentation: Any, crop_box: CropBox) -> Any:
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

def process_image_task(args: tuple):
    (
        image_info,
        annotations,
        image_dir,
        padding,
        min_pixels_area,
        categories,
        category_id_remap
    ) = args

    image_path = Path(image_dir) / image_info["file_name"]

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    img_h, img_w = img.shape[:2]

    results = []
    ignored = []

    for ann in annotations:
        area = ann.get("area", 0)

        if area < min_pixels_area:
            ignored.append(
                {
                    "image": image_info.get("file_name"),
                    "object_id": ann.get("id"),
                    "area": area,
                }
            )
            continue

        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"Invalid bbox for annotation {ann.get("id")}")
        
        crop_box = _compute_crop_box(
            bbox=bbox,
            img_h=img_h,
            img_w=img_w,
            padding=padding
        )

        crop = img[crop_box.y0:crop_box.y1, crop_box.x0:crop_box.x1]

        output_image_name = _build_output_name(
            image_info["file_name"],
            ann.get("id"),
            ann.get("category_id"),
            image_path.suffix
        )

        crop_json = _build_crop_json(
            image_info=image_info,
            annotation=ann,
            categories=categories,
            crop_box=crop_box,
            output_image_name=output_image_name,
            original_image_size=(img_w, img_h),
            category_id_remap=category_id_remap
        )

        results.append((
            crop,
            output_image_name,
            crop_json
        ))
    
    return results, ignored

def writer_worker(queue: Queue, image_dir: Path, json_dir: Path):
    while True:
        item = queue.get()
        if item is None:
            break

        crop, image_name, crop_json = item

        image_path = str(image_dir / image_name)
        json_path = json_dir / (Path(image_name).stem + ".json")

        cv2.imwrite(image_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        with open(json_path, "w") as f:
            json.dump(crop_json, f)

        queue.task_done()


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

    This class processes one image directory and one COCO JSON file. Use the
    CLI or config to run it across multiple directory/JSON pairs.

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

        categories, category_id_remap = self._normalize_categories(
            coco.get("categories", [])
        )

        self.crop_images_output_dir.mkdir(parents=True, exist_ok=True)
        self.crop_json_output_dir.mkdir(parents=True, exist_ok=True)

        annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}

        for ann in coco["annotations"]:
            annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        queue = Queue(maxsize=1000)

        writer = Thread(
            target=writer_worker,
            args=(queue, self.crop_images_output_dir, self.crop_json_output_dir),
            daemon=True
        )

        writer.start()

        tasks = [
            (
                images_by_id[image_id],
                anns,
                str(self.image_dir),
                self.padding,
                self.min_pixels_area,
                categories,
                category_id_remap
            )    
            for image_id, anns in annotations_by_image.items()
        ]

        ignored_rows = []

        with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            futures = [executor.submit(process_image_task, t) for t in tasks]

            with tqdm(total=len(futures)) as pbar:
                for future in as_completed(futures):
                    results, ignored = future.result()
                    ignored_rows.extend(ignored)

                    for item in results:
                        queue.put(item)

                    pbar.update(1)
            
        queue.join()
        queue.put(None)

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
