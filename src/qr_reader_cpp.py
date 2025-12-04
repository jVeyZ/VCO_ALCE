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
DEFAULT_JSON_DIR = BASE_DIR / "img" / "output_json"
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
		"-o",
		"--output-json-dir",
		type=Path,
		default=DEFAULT_JSON_DIR,
		help=f"Directory to write decoded QR JSON files (default: {DEFAULT_JSON_DIR})",
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
	# Default: extract fields from QR payload
	parser.add_argument(
		"--no-extract-fields",
		action="store_false",
		dest="extract_fields",
		default=True,
		help=(
			"Disable parsing of the last two ';'-separated fields from the QR payload."
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


def save_results_json(image_path: Path, results: List, output_dir: Path, extract_fields: bool) -> None:
	"""Save decoded QR results to a JSON file named after the image."""
	try:
		import json
		output_dir.mkdir(parents=True, exist_ok=True)
		data = {
			"image": str(image_path),
			"name": image_path.stem,
			"results": [],
		}
		for result in results:
			text = getattr(result, "text", "")
			formatted = build_payload_output(text, extract_fields)
			format_attr = getattr(result, "format", None)
			format_name = getattr(format_attr, "name", str(format_attr)) if format_attr else "unknown"
			# Structured parsing of last two fields when enabled
			numeric_count = None
			numeric_indices = None
			if extract_fields:
				parsed = extract_last_fields(normalize_payload(text))
				if parsed:
					num_field, fill_field = parsed
					numeric_count = int(num_field) if num_field.isdigit() else None
					# fill_field like "1,2,5" -> list[int]
					numeric_indices = [int(x) for x in fill_field.split(",") if x.isdigit()]
			data["results"].append({
				"format": format_name,
				"raw": normalize_payload(text),
				"formatted": formatted,
				"numeric_count": numeric_count,
				"numeric_indices": numeric_indices,
			})
		out_path = output_dir / f"{image_path.stem}.json"
		with open(out_path, "w", encoding="utf-8") as f:
			json.dump(data, f, ensure_ascii=False, indent=2)
	except Exception as exc:
		print(f"[WARN] Failed to write JSON for {image_path}: {exc}")


def scan_images(image_paths: Iterable[Path], extract_fields: bool, output_dir: Path) -> None:
	for image_path in image_paths:
		try:
			results = decode_barcodes(image_path)
		except RuntimeError as exc:
			print(f"[WARN] {exc}")
			continue
		if not results:
			print(f"{image_path}: no QR codes found")
			# still write an empty file for traceability
			save_results_json(image_path, [], output_dir, extract_fields)
			continue
		for result in results:
			text = getattr(result, "text", "")
			formatted_payload = build_payload_output(text, extract_fields)
			format_attr = getattr(result, "format", None)
			format_name = getattr(format_attr, "name", str(format_attr)) if format_attr else "unknown"
			print(f"{image_path}: [{format_name}] {formatted_payload}")
		# write JSON summary per image
		save_results_json(image_path, results, output_dir, extract_fields)


def scan_images_quiet(image_paths: Iterable[Path], extract_fields: bool, output_dir: Path) -> None:
	for image_path in image_paths:
		try:
			results = decode_barcodes(image_path)
		except RuntimeError as exc:
			print(f"[WARN] {exc}")
			continue
		if not results:
			# still write an empty file for traceability
			save_results_json(image_path, [], output_dir, extract_fields)
			continue
		payloads = []
		for result in results:
			text = getattr(result, "text", "")
			formatted_payload = build_payload_output(text, extract_fields)
			format_attr = getattr(result, "format", None)
			format_name = getattr(format_attr, "name", str(format_attr)) if format_attr else "unknown"
			payloads.append(f"[{format_name}] {formatted_payload}")
		print(f"{image_path}: {', '.join(payloads)}")
		# write JSON summary per image
		save_results_json(image_path, results, output_dir, extract_fields)


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
		scan_images_quiet(images, args.extract_fields, args.output_json_dir)
	else:
		scan_images(images, args.extract_fields, args.output_json_dir)


if __name__ == "__main__":
	main()
