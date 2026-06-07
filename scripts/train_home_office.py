import argparse
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


def resolve_dataset_yaml(data_path: str) -> str:
    path = Path(data_path)
    repo_root = Path(__file__).resolve().parents[1]
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()

    config = yaml.safe_load(path.read_text())
    dataset_root = Path(config.get("path", ""))
    if not dataset_root.is_absolute():
        candidates = [
            (repo_root / dataset_root).resolve(),
            (path.parent / dataset_root).resolve(),
            (repo_root / "datasets/home_office").resolve(),
        ]
        dataset_root = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])

    config["path"] = str(dataset_root)

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(config, tmp, sort_keys=False)
    tmp.close()
    print(f"Using dataset root: {dataset_root}")
    return tmp.name


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for home/office obstacle classes.")
    parser.add_argument("--data", default="training/home_office_dataset.yaml")
    parser.add_argument("--base", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="home_office_yolo")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    model = YOLO(args.base)
    data_yaml = resolve_dataset_yaml(args.data)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        workers=args.workers,
        patience=15,
        cos_lr=True,
        close_mosaic=10,
        degrees=3.0,
        translate=0.08,
        scale=0.35,
        fliplr=0.4,
    )


if __name__ == "__main__":
    main()
