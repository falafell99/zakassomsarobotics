# Home/Office Training Notes

Use this when the base COCO YOLO model is not enough for home and office scenes.

## 1. Collect Images

Collect frames from laptop camera:

```bash
python scripts/collect_frames.py --out datasets/home_office/raw --every 1.0
```

Capture varied examples:

- home hallway, room corners, walls, doorways
- office desks, chairs, monitor, keyboard, mouse
- shelves, cabinets, boxes, trash cans
- low obstacles such as rugs, bags, boxes
- stairs, individual steps, railings, handrails
- small non-critical items: cables, chargers, paper, notebooks
- lighting variations: day, night, backlight
- close / medium / far distances

## 2. Annotate

Use Label Studio, Roboflow, CVAT, or LabelImg.

Export labels in YOLO format:

```text
datasets/home_office/images/train/*.jpg
datasets/home_office/labels/train/*.txt
datasets/home_office/images/val/*.jpg
datasets/home_office/labels/val/*.txt
```

Classes are defined in `training/home_office_dataset.yaml`.

For this project, prioritize:

- wall
- door
- doorway
- stairs
- step
- railing / handrail
- chair / office_chair
- desk / table
- sofa / bed
- cabinet / wardrobe / bookshelf
- monitor / laptop / keyboard / mouse
- box / storage_box / laundry_basket
- trash_can / plant
- cable / charger / paper / notebook as low-priority context

## 3. Train

```bash
python scripts/train_home_office.py --data training/home_office_dataset.yaml --base yolo11n.pt --epochs 80 --imgsz 640
```

Best weights will be under:

```text
runs/detect/home_office_yolo/weights/best.pt
```

Then set `.env`:

```bash
YOLO_MODEL=runs/detect/home_office_yolo/weights/best.pt
```
