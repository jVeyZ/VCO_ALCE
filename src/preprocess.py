from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import cv2
import numpy as np


#Work directory:
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "img" / "vco_test"
DEFAULT_OUTPUT_DIR = BASE_DIR / "img" / "output"

# A4 @ 300 DPI in portrait and landscape orientations.
PORTRAIT_A4 = (2480, 3508)  # (width, height)
LANDSCAPE_A4 = (3508, 2480)
DOT_MIN_AREA = 60.0
DOT_MAX_AREA = 6000.0


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Warp scans using four calibration dots so the output fills an A4 sheet, "
			"then binarize the result."
		)
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=DEFAULT_INPUT_DIR,
		help=f"Folder with input scans (default: {DEFAULT_INPUT_DIR})",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help=f"Folder to write processed scans (default: {DEFAULT_OUTPUT_DIR})",
	)
	parser.add_argument(
		"--orientation",
		choices=["auto", "portrait", "landscape"],
		default="auto",
		help=(
			"Portrait stays 2480x3508, landscape 3508x2480, auto matches the input orientation."
		),
	)
	parser.add_argument(
		"--preview-only",
		action="store_true",
		help="Only load the input images and preview them without further processing.",
	)
	parser.add_argument(
		"--show-dots",
		action="store_true",
		help="Display detected calibration dots before warping each page.",
	)
	parser.add_argument(
		"--show-results",
		action="store_true",
		help="Display the warped and binarized page before saving it.",
	)
	return parser.parse_args()


def ensure_directory(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def gather_images(path: Path) -> List[Path]:
	exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
	return sorted(p for p in path.iterdir() if p.suffix.lower() in exts)


def preview_inputs(image_paths: Iterable[Path]) -> None:
	print("Preview mode: press any key to advance, ESC/Q to exit.")
	for image_path in image_paths:
		image = cv2.imread(str(image_path))
		if image is None:
			print(f"[WARN] Could not read image: {image_path}")
			continue
		cv2.imshow(f"preview::{image_path.name}", image)
		key = cv2.waitKey(0) & 0xFF
		cv2.destroyWindow(f"preview::{image_path.name}")
		if key in (27, ord("q")):
			break
	cv2.destroyAllWindows()


def show_calibration_dots(image: np.ndarray, points: np.ndarray, label: str) -> None:
	annotated = image.copy()
	point_color = (0, 255, 0)
	text_color = (0, 0, 255)
	for idx, (x, y) in enumerate(points):
		center = (int(round(x)), int(round(y)))
		cv2.circle(annotated, center, 15, point_color, 3)
		cv2.putText(
			annotated,
			f"{idx}",
			(center[0] + 10, center[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.9,
			text_color,
			2,
			cv2.LINE_AA,
		)
	cv2.polylines(
		annotated,
		[np.int32(points).reshape((-1, 1, 2))],
		isClosed=True,
		color=(255, 0, 0),
		thickness=2,
	)
	window_name = f"dots::{label}"
	cv2.imshow(window_name, annotated)
	key = cv2.waitKey(0) & 0xFF
	cv2.destroyWindow(window_name)
	if key in (27, ord("q")):
		raise KeyboardInterrupt("Calibration preview cancelled by user")


def show_processed_image(image: np.ndarray, label: str) -> None:
	window_name = f"result::{label}"
	cv2.imshow(window_name, image)
	key = cv2.waitKey(0) & 0xFF
	cv2.destroyWindow(window_name)
	if key in (27, ord("q")):
		raise KeyboardInterrupt("Result preview cancelled by user")


def detect_calibration_dots(gray: np.ndarray) -> np.ndarray:
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	_, thresh = cv2.threshold(
		blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
	)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
	contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	centers: List[Tuple[float, float]] = []
	areas: List[float] = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if area < DOT_MIN_AREA or area > DOT_MAX_AREA:
			continue
		perimeter = cv2.arcLength(contour, True)
		if perimeter == 0:
			continue
		circularity = 4 * np.pi * area / (perimeter * perimeter)
		if circularity < 0.75:
			continue
		moments = cv2.moments(contour)
		if moments["m00"] == 0:
			continue
		cx = float(moments["m10"] / moments["m00"])
		cy = float(moments["m01"] / moments["m00"])
		centers.append((cx, cy))
		areas.append(area)

	if len(centers) < 4:
		raise RuntimeError(
			f"Expected 4 calibration dots, but only found {len(centers)}."
		)

	idxs = np.argsort(areas)[-4:]
	selected = np.array([centers[i] for i in idxs], dtype=np.float32)
	return order_points(selected)


def order_points(points: np.ndarray) -> np.ndarray:
	if points.shape != (4, 2):
		raise ValueError(f"Need exactly four points, received shape {points.shape}.")
	s = points.sum(axis=1)
	diff = np.diff(points, axis=1)
	ordered = np.zeros((4, 2), dtype=np.float32)
	ordered[0] = points[np.argmin(s)]  # top-left
	ordered[2] = points[np.argmax(s)]  # bottom-right
	ordered[1] = points[np.argmin(diff)]  # top-right
	ordered[3] = points[np.argmax(diff)]  # bottom-left
	return ordered


def compute_target_size(image_shape: Sequence[int], orientation: str) -> Tuple[int, int]:
	if orientation == "portrait":
		return PORTRAIT_A4
	if orientation == "landscape":
		return LANDSCAPE_A4
	height, width = image_shape[:2]
	return LANDSCAPE_A4 if width > height else PORTRAIT_A4


def warp_document(
	image: np.ndarray, source_points: np.ndarray, target_size: Tuple[int, int]
) -> np.ndarray:
	target_w, target_h = target_size
	destination = np.array(
		[[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
		dtype=np.float32,
	)
	matrix = cv2.getPerspectiveTransform(source_points, destination)
	return cv2.warpPerspective(image, matrix, target_size)


def binarize(image: np.ndarray) -> np.ndarray:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cleaned = cv2.medianBlur(gray, 3)
	return cv2.adaptiveThreshold(
		cleaned,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY,
		35,
		10,
	)


def process_image(
	image_path: Path,
	output_dir: Path,
	orientation: str,
	show_dots: bool,
	show_results: bool,
) -> Path:
	image = cv2.imread(str(image_path))
	if image is None:
		raise RuntimeError(f"Could not read image: {image_path}")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	source_points = detect_calibration_dots(gray)
	if show_dots:
		show_calibration_dots(image, source_points, image_path.name)
	target_size = compute_target_size(image.shape, orientation)
	warped = warp_document(image, source_points, target_size)
	binary = binarize(warped)
	if show_results:
		show_processed_image(binary, image_path.name)
	output_name = f"processed_{image_path.name}"
	output_path = output_dir / output_name
	cv2.imwrite(str(output_path), binary)
	return output_path


def main() -> None:
	args = parse_args()
	if not args.input_dir.exists():
		raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
	ensure_directory(args.output_dir)

	images = gather_images(args.input_dir)
	if not images:
		raise FileNotFoundError(
			f"No images found in {args.input_dir}. Expected formats: PNG, JPG, TIF, BMP."
		)

	if args.preview_only:
		preview_inputs(images)
		return

	try:
		for image_path in images:
			try:
				output_path = process_image(
					image_path,
					args.output_dir,
					args.orientation,
					args.show_dots,
					args.show_results,
				)
				print(f"Saved {output_path}")
			except Exception as exc:  # noqa: BLE001 - keep processing other files
				print(f"[WARN] Skipped {image_path}: {exc}")
	except KeyboardInterrupt:
		print("Aborted by user during calibration preview.")


if __name__ == "__main__":
	main()
