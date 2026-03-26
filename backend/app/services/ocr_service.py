"""
OCR Service — extracts text from job posting images using Tesseract.
Passes extracted text into the scam detection pipeline.
"""

import os
import io
import logging
from typing import Union

import pytesseract
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)

# Set Tesseract path for Windows if needed
_TESSERACT_PATH = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if os.path.isfile(_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_PATH


# ---------------------------------------------------------------------------
# Image preprocessing — improves OCR accuracy on noisy job screenshots
# ---------------------------------------------------------------------------

def _preprocess_image(img: Image.Image) -> Image.Image:
    """Convert to grayscale, sharpen, and increase contrast."""
    img = img.convert("L")                              # grayscale
    img = img.filter(ImageFilter.SHARPEN)               # sharpen edges
    img = ImageEnhance.Contrast(img).enhance(2.0)       # boost contrast
    return img


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_image(source: Union[str, bytes, Image.Image]) -> dict:
    """
    Extract text from a job posting image.

    Args:
        source: File path (str), raw bytes, or PIL Image object

    Returns:
        {
            "text":       str,   # extracted text
            "char_count": int,
            "success":    bool,
            "error":      str | None
        }
    """
    try:
        if isinstance(source, str):
            img = Image.open(source)
        elif isinstance(source, bytes):
            img = Image.open(io.BytesIO(source))
        elif isinstance(source, Image.Image):
            img = source
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        img = _preprocess_image(img)

        # PSM 6 = assume a single uniform block of text (good for job postings)
        config = "--psm 6 --oem 3"
        text = pytesseract.image_to_string(img, config=config)
        text = text.strip()

        return {
            "text":       text,
            "char_count": len(text),
            "success":    True,
            "error":      None,
        }

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return {
            "text":       "",
            "char_count": 0,
            "success":    False,
            "error":      str(e),
        }


def extract_text_from_upload(file_bytes: bytes, filename: str = "") -> dict:
    """
    Wrapper for FastAPI UploadFile bytes.
    Validates file type before processing.
    """
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    ext = os.path.splitext(filename.lower())[1] if filename else ""

    if ext and ext not in allowed:
        return {
            "text": "", "char_count": 0, "success": False,
            "error": f"Unsupported file type: {ext}. Allowed: {allowed}",
        }

    return extract_text_from_image(file_bytes)
