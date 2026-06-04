"""
Image preprocessing utilities for smart blind-assistance glasses backend.
Handles decoding, resizing, and re-encoding of base64 JPEG images.
"""

import base64
import io
from PIL import Image
from typing import Optional


def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decode a base64 string to a PIL Image.

    Args:
        base64_string: Base64 encoded JPEG image string

    Returns:
        PIL Image object
    """
    # Remove data URL prefix if present
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]

    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_string)

    # Open as PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if necessary (handles RGBA, P, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def resize_image(image: Image.Image, max_width: int = 800, max_height: int = 600) -> Image.Image:
    """
    Resize image to fit within max dimensions while maintaining aspect ratio.

    Args:
        image: PIL Image object
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels

    Returns:
        Resized PIL Image object
    """
    # Get original dimensions
    width, height = image.size

    # Check if resize is needed
    if width <= max_width and height <= max_height:
        return image

    # Calculate scaling factor
    width_ratio = max_width / width
    height_ratio = max_height / height
    scale_factor = min(width_ratio, height_ratio)

    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize image with high-quality resampling
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return resized_image


def encode_image_to_base64(image: Image.Image, quality: int = 70) -> str:
    """
    Encode a PIL Image to base64 JPEG string.

    Args:
        image: PIL Image object
        quality: JPEG quality (1-100, lower = smaller file)

    Returns:
        Base64 encoded JPEG string (without data URL prefix)
    """
    # Save image to bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)

    # Encode to base64
    base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return base64_string


def process_image(base64_image: str, max_width: int = 800, max_height: int = 600, quality: int = 70) -> str:
    """
    Full image processing pipeline: decode → resize → re-encode.

    Args:
        base64_image: Base64 encoded JPEG image string
        max_width: Maximum width for resize
        max_height: Maximum height for resize
        quality: JPEG quality for re-encoding

    Returns:
        Processed base64 encoded JPEG string
    """
    # Decode base64 to PIL Image
    image = decode_base64_image(base64_image)

    # Resize if necessary
    image = resize_image(image, max_width, max_height)

    # Re-encode to base64 JPEG
    processed_base64 = encode_image_to_base64(image, quality)

    return processed_base64


def get_image_size_kb(base64_string: str) -> float:
    """
    Calculate the size of a base64 image in kilobytes.

    Args:
        base64_string: Base64 encoded image string

    Returns:
        Size in kilobytes
    """
    # Remove data URL prefix if present
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]

    # Calculate size
    num_bytes = len(base64_string) * 3 / 4 - base64_string.count("=", -2)
    size_kb = num_bytes / 1024

    return size_kb
