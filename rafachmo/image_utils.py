import base64
import io
from PIL import Image
import numpy as np


def decode_base64_image(base64_image: str) -> Image.Image:
    if "base64," in base64_image:
        base64_image = base64_image.split("base64,", 1)[1]
    data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(data))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image)
    return rgb[:, :, ::-1].copy()


def normalize_bbox(box: tuple[int, int, int, int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [
        int(max(0, min(1000, x1 / width * 1000))),
        int(max(0, min(1000, y1 / height * 1000))),
        int(max(0, min(1000, x2 / width * 1000))),
        int(max(0, min(1000, y2 / height * 1000))),
    ]

