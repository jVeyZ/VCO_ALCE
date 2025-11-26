"""Extract ID code from a fixed physical region of processed A4 scans."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2

PYTESSERACT_ERROR = None
try:
	import pytesseract  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - optional dependency
	pytesseract = None
	PYTESSERACT_ERROR = exc


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "img" / "output"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Physical dimensions in centimeters for an A4 page.
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7

# Bounding box coordinates in centimeters, top-left origin.
BOUNDING_BOX_CM = (0.25, 8.45, 10.25, 9.90)  # (x1, y1, x2, y2)

# Regex for an 8-digit sequence followed by a letter on a compacted string.
PATTERN = re.compile(r"(\d{8})([A-Z])")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Crop the predefined physical region from each processed scan and extract an 8-digit code followed by a letter."
		)
	)
	parser.add_argument(
		"-i",
		"--input-dir",
		type=Path,
		default=DEFAULT_INPUT_DIR,
		help=f"Directory with processed scans (default: {DEFAULT_INPUT_DIR})",
	)
	parser.add_argument(
		"--tesseract-lang",
		default="eng",
		help="Language(s) to pass to Tesseract (default: eng).",
	)
	parser.add_argument(
		"-s",
		"--show-roi",
		action="store_true",
		help="Display the computed ROI for each image before running OCR.",
	)
	parser.add_argument(
		"--debug-text",
		action="store_true",
		help="Print the normalized text used for pattern matching.",
	)
	return parser.parse_args()


def gather_images(path: Path) -> Iterable[Path]:
	return sorted(
		candidate
		for candidate in path.iterdir()
		if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTS
	)


def load_image(image_path: Path):
	image = cv2.imread(str(image_path))
	if image is None:
		raise RuntimeError(f"Could not read image: {image_path}")
	return image


def crop_physical_roi(image, bbox_cm: Tuple[float, float, float, float]):
	height, width = image.shape[:2]
	x1, y1, x2, y2 = bbox_cm
	left = int(round((x1 / A4_WIDTH_CM) * width))
	right = int(round((x2 / A4_WIDTH_CM) * width))
	top = int(round((y1 / A4_HEIGHT_CM) * height))
	bottom = int(round((y2 / A4_HEIGHT_CM) * height))
	left, right = max(0, left), min(width, right)
	top, bottom = max(0, top), min(height, bottom)
	if right <= left or bottom <= top:
		raise ValueError("Invalid ROI after conversion; check bounding box coordinates.")
	return image[top:bottom, left:right]


def preprocess_roi(roi):
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
	blurred = cv2.GaussianBlur(scaled, (3, 3), 0)
	_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
	return cleaned


def ensure_tesseract() -> None:
	if pytesseract is None:  # pragma: no cover - optional dependency path
		raise ImportError(
			"pytesseract is required. Install it with 'pip install pytesseract' and ensure Tesseract OCR is available."
		) from PYTESSERACT_ERROR


def run_ocr(image, lang: str) -> str:
	ensure_tesseract()
	config = (
		"--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	)  # single line alphanumeric only
	text = pytesseract.image_to_string(image, lang=lang, config=config)
	return text.strip()


def normalize_text(text: str) -> str:
	replaced = text.upper()
	replaced = replaced.replace("O", "0").replace("I", "1").replace("L", "1")
	compacted = re.sub(r"[^0-9A-Z]", "", replaced)
	return compacted


def extract_code(normalized_text: str) -> Optional[Tuple[str, str]]:
	match = PATTERN.search(normalized_text)
	if not match:
		return None
	digits, letter = match.groups()
	return digits, letter


def process_image(image_path: Path, lang: str, show_roi: bool, debug_text: bool) -> None:
	image = load_image(image_path)
	roi = crop_physical_roi(image, BOUNDING_BOX_CM)
	if show_roi:
		cv2.imshow(f"roi::{image_path.name}", roi)
		key = cv2.waitKey(0) & 0xFF
		cv2.destroyWindow(f"roi::{image_path.name}")
		if key in (27, ord("q")):
			raise KeyboardInterrupt("ROI preview cancelled by user")
	prepared = preprocess_roi(roi)
	text = run_ocr(prepared, lang)
	compacted = normalize_text(text)
	if debug_text:
		print(f"{image_path.name}: normalized='{compacted}' raw='{text}'")
	result = extract_code(compacted)
	if result:
		digits, letter = result
		print(f"{image_path.name}: code={digits}{letter} (raw='{text}')")
	else:
		print(f"{image_path.name}: no match (raw='{text}')")


def main() -> None:
	args = parse_args()
	if not args.input_dir.exists():
		raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

	images = list(gather_images(args.input_dir))
	if not images:
		raise FileNotFoundError(
			f"No images found in {args.input_dir}. Expected formats: {', '.join(sorted(SUPPORTED_EXTS))}"
		)

	try:
		for image_path in images:
			try:
				process_image(image_path, args.tesseract_lang, args.show_roi, args.debug_text)
			except Exception as exc:  # noqa: BLE001
				print(f"[WARN] Failed to process {image_path}: {exc}")
	except KeyboardInterrupt:
		print("Aborted by user during ROI preview.")


if __name__ == "__main__":
	main()
