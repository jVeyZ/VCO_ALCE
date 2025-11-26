"""Scan processed images for QR codes using ZXing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2

ZXING_IMPORT_ERROR = None
try:
	import zxingcpp
except ImportError as exc:  # pragma: no cover - handled at runtime
	zxingcpp = None
	ZXING_IMPORT_ERROR = exc


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "img" / "output"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
NUMBER_LABEL = "Número de preguntas"
FILL_LABEL = "Preguntas de rellenar"
LAST_FIELDS_PATTERN = re.compile(r"([^;]+);([^;]+)\s*$")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Scan every image in the output folder for QR codes using ZXing"
	)
	parser.add_argument(
		"-i",
		"--input-dir",
		type=Path,
		default=DEFAULT_INPUT_DIR,
		help=f"Directory with processed scans (default: {DEFAULT_INPUT_DIR})",
	)
	parser.add_argument(
		"-r",
		"--recursive",
		action="store_true",
		help="Scan subfolders recursively instead of only the top-level directory.",
	)
	parser.add_argument(
		"-q",
		"--quiet",
		action="store_true",
		help="Only print files with at least one QR code detected.",
	)
	parser.add_argument(
		"-e",
		"--extract-fields",
		action="store_true",
		help=(
			"Parse the QR payload to extract the last two ';'-separated fields and label them."
		),
	)
	return parser.parse_args()


def gather_images(path: Path, recursive: bool) -> List[Path]:
	if recursive:
		candidates = path.rglob("*")
	else:
		candidates = path.iterdir()
	return sorted(
		candidate
		for candidate in candidates
		if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTS
	)


def ensure_zxingcpp() -> None:
	if zxingcpp is None:  # pragma: no cover - dependency error path
		raise ImportError(
			"zxing-cpp bindings are required. Install them with 'pip install zxing-cpp'."
		) from ZXING_IMPORT_ERROR

def normalize_payload(value) -> str:
	if isinstance(value, bytes):
		for encoding in ("utf-8", "latin-1"):
			try:
				return value.decode(encoding)
			except UnicodeDecodeError:
				continue
		return value.decode("utf-8", errors="replace")
	if value is None:
		return ""
	return str(value)


def load_image(image_path: Path):
	image = cv2.imread(str(image_path))
	if image is None:
		raise RuntimeError(f"Failed to load image: {image_path}")
	return image


def decode_barcodes(image_path: Path) -> List:
	ensure_zxingcpp()
	try:
		image = load_image(image_path)
		return list(zxingcpp.read_barcodes(image))
	except Exception as exc:  # noqa: BLE001
		raise RuntimeError(f"Failed to decode {image_path}: {exc}") from exc


def extract_last_fields(payload: str) -> Optional[Tuple[str, str]]:
	cleaned = payload.strip()
	match = LAST_FIELDS_PATTERN.search(cleaned)
	if not match:
		return None
	first, second = match.groups()
	first_number = "".join(re.findall(r"\d", first))
	second_numbers = re.findall(r"\d+", second)
	if not first_number or not second_numbers:
		return None
	return first_number, ",".join(second_numbers)


def build_payload_output(payload: str, extract_fields: bool) -> str:
	text = normalize_payload(payload)
	if not extract_fields:
		return text
	parsed = extract_last_fields(text)
	if not parsed:
		return text
	num_field, fill_field = parsed
	return (
		f"{text} | {NUMBER_LABEL}: {num_field} | {FILL_LABEL}: {fill_field}"
	)


def scan_images(image_paths: Iterable[Path], extract_fields: bool) -> None:
	for image_path in image_paths:
		try:
			results = decode_barcodes(image_path)
		except RuntimeError as exc:
			print(f"[WARN] {exc}")
			continue
		if not results:
			print(f"{image_path}: no QR codes found")
			continue
		for result in results:
			text = getattr(result, "text", "")
			formatted_payload = build_payload_output(text, extract_fields)
			format_attr = getattr(result, "format", None)
			format_name = getattr(format_attr, "name", str(format_attr)) if format_attr else "unknown"
			print(f"{image_path}: [{format_name}] {formatted_payload}")


def scan_images_quiet(image_paths: Iterable[Path], extract_fields: bool) -> None:
	for image_path in image_paths:
		try:
			results = decode_barcodes(image_path)
		except RuntimeError as exc:
			print(f"[WARN] {exc}")
			continue
		if not results:
			continue
		payloads = []
		for result in results:
			text = getattr(result, "text", "")
			formatted_payload = build_payload_output(text, extract_fields)
			format_attr = getattr(result, "format", None)
			format_name = getattr(format_attr, "name", str(format_attr)) if format_attr else "unknown"
			payloads.append(f"[{format_name}] {formatted_payload}")
		print(f"{image_path}: {', '.join(payloads)}")


def main() -> None:
	args = parse_args()
	if not args.input_dir.exists():
		raise FileNotFoundError(
			f"Input directory does not exist: {args.input_dir}. Run preprocess.py first."
		)

	images = gather_images(args.input_dir, args.recursive)
	if not images:
		raise FileNotFoundError(
			f"No images found in {args.input_dir}. Expected formats: {', '.join(sorted(SUPPORTED_EXTS))}"
		)

	if args.quiet:
		scan_images_quiet(images, args.extract_fields)
	else:
		scan_images(images, args.extract_fields)


if __name__ == "__main__":
	main()
