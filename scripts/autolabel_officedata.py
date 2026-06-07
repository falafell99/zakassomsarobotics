import argparse
import random
import shutil
import subprocess
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


PROJECT_CLASS_IDS = {
    "person": 0,
    "chair": 5,
    "dining table": 8,
    "couch": 11,
    "bed": 12,
    "tv": 18,
    "laptop": 19,
    "keyboard": 20,
    "mouse": 21,
    "cell phone": 22,
    "potted plant": 26,
    "backpack": 33,
    "suitcase": 34,
    "cat": 35,
    "dog": 35,
    "bottle": 30,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}


def convert_to_jpg(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(src, dst)
        return True

    try:
        image = Image.open(src)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(dst, "JPEG", quality=92)
        return True
    except Exception:
        pass

    if src.suffix.lower() in {".heic", ".heif"}:
        try:
            subprocess.run(
                ["sips", "-s", "format", "jpeg", str(src), "--out", str(dst)],
                check=True,
                capture_output=True,
            )
            return dst.exists() and dst.stat().st_size > 0
        except Exception:
            return False

    return False


def collect_images(source: Path) -> list[Path]:
    return sorted(
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def split_images(images: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def write_labels(model: YOLO, image_paths: list[Path], label_dir: Path, conf: float, imgsz: int) -> int:
    label_dir.mkdir(parents=True, exist_ok=True)
    total_boxes = 0

    predictions = model.predict(
        source=[str(path) for path in image_paths],
        conf=conf,
        imgsz=imgsz,
        stream=True,
        verbose=False,
    )

    for image_path, result in zip(image_paths, predictions):
        label_path = label_dir / f"{image_path.stem}.txt"
        rows = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            raw_label = str(model.names.get(cls_id, cls_id)).lower()
            target_id = PROJECT_CLASS_IDS.get(raw_label)
            if target_id is None:
                continue
            x, y, w, h = [float(v) for v in box.xywhn[0].tolist()]
            rows.append(f"{target_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        label_path.write_text("\n".join(rows) + ("\n" if rows else ""))
        total_boxes += len(rows)

    return total_boxes


def prepare_split(split_name: str, images: list[Path], out_root: Path) -> list[Path]:
    image_dir = out_root / "images" / split_name
    label_dir = out_root / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    prepared = []
    for index, src in enumerate(images, start=1):
        safe_stem = f"{split_name}_{index:04d}_{src.stem.replace(' ', '_')}"
        dst = image_dir / f"{safe_stem}.jpg"
        if convert_to_jpg(src, dst):
            prepared.append(dst)
        else:
            print(f"[skip] could not convert: {src}")
    return prepared


def main():
    parser = argparse.ArgumentParser(description="Prepare officedata and auto-label common COCO classes for YOLO training.")
    parser.add_argument("--source", default="/Users/ax1le/Downloads/officedata")
    parser.add_argument("--out", default="datasets/home_office")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    out_root = Path(args.out)

    images = collect_images(source)
    if not images:
        raise SystemExit(f"No images found in {source}")

    train_src, val_src = split_images(images, args.val_ratio, args.seed)
    train_images = prepare_split("train", train_src, out_root)
    val_images = prepare_split("val", val_src, out_root)

    model = YOLO(args.model)
    train_boxes = write_labels(model, train_images, out_root / "labels" / "train", args.conf, args.imgsz)
    val_boxes = write_labels(model, val_images, out_root / "labels" / "val", args.conf, args.imgsz)

    print("Auto-label complete")
    print(f"source images: {len(images)}")
    print(f"train images:  {len(train_images)} | boxes: {train_boxes}")
    print(f"val images:    {len(val_images)} | boxes: {val_boxes}")
    print(f"dataset root:  {out_root.resolve()}")
    print()
    print("Important: review labels before training, especially walls, doors, stairs, railings, and handrails.")


if __name__ == "__main__":
    main()
