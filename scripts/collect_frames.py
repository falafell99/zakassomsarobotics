import argparse
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Collect home/office frames for YOLO annotation.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out", default="datasets/home_office/raw")
    parser.add_argument("--every", type=float, default=1.0, help="Seconds between saved frames.")
    parser.add_argument("--prefix", default="home_office")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    last = 0.0
    count = len(list(out_dir.glob("*.jpg")))

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        now = time.time()
        if now - last >= args.every:
            last = now
            count += 1
            path = out_dir / f"{args.prefix}_{count:05d}.jpg"
            cv2.imwrite(str(path), frame)
            print(path)

        cv2.imshow("collect frames - Q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
