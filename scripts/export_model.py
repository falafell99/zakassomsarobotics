import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export trained YOLO model.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "coreml"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="Export in FP16 (half precision)")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.export(format=args.format, imgsz=args.imgsz, half=args.half)


if __name__ == "__main__":
    main()
