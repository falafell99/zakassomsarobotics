import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for home/office obstacle classes.")
    parser.add_argument("--data", default="training/home_office_dataset.yaml")
    parser.add_argument("--base", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="home_office_yolo")
    args = parser.parse_args()

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=20,
        cos_lr=True,
        close_mosaic=10,
        degrees=3.0,
        translate=0.08,
        scale=0.35,
        fliplr=0.4,
    )


if __name__ == "__main__":
    main()
