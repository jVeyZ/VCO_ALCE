"""Segment DNI and answer-area digits for manual labeling.

Usage:
  python segment_digits.py <image_path> [mode]
Modes:
  full (default) - crops DNI region + answer area digits (best-effort)
  dni            - only DNI region segmentation
  raw            - whole image segmentation (legacy)

Outputs JSON to stdout: {width, height, segments: [{index,bbox,aspect,image,kind}]}"""
from __future__ import annotations

import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# Physical dimensions in centimeters for an A4 page.
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7

# DNI bounding box in cm (x1, y1, x2, y2) from process.py
DNI_BOX_CM = (0.25, 8.45, 10.25, 9.90)

# Answer area (bubbles) rough region in cm, covering both columns
ANSWER_BOX_CM = (1.5, 11.5, 19.5, 27.5)


@dataclass
class Segment:
	kind: str
	bbox: Tuple[int, int, int, int]
	image_b64: str
	aspect: float


def encode_png(array: np.ndarray) -> str:
	_, buf = cv2.imencode(".png", array)
	return base64.b64encode(buf).decode("ascii")


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


def preprocess_roi_for_segmentation(roi):
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
	blurred = cv2.GaussianBlur(scaled, (5, 5), 0)
	thresh = cv2.adaptiveThreshold(
		blurred,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY_INV,
		11,
		2,
	)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
	return cleaned, scaled


def segment_characters(binary_image, original_gray) -> List[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
	contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	height, width = binary_image.shape
	char_candidates = []
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		if h > height * 0.2 and h < height * 0.9 and w > 2:
			char_candidates.append((x, y, w, h))
	char_candidates.sort(key=lambda c: c[0])
	if char_candidates:
		median_y = np.median([c[1] for c in char_candidates])
		char_candidates = [c for c in char_candidates if abs(c[1] - median_y) < height * 0.3]
	final_candidates = []
	if char_candidates:
		median_w = np.median([c[2] for c in char_candidates])
		for x, y, w, h in char_candidates:
			if w > median_w * 1.8:
				half_w = w // 2
				final_candidates.append((x, y, half_w, h))
				final_candidates.append((x + half_w, y, w - half_w, h))
			else:
				final_candidates.append((x, y, w, h))
		char_candidates = final_candidates
	chars = []
	for x, y, w, h in char_candidates:
		pad = 4
		roi = binary_image[max(0, y - pad): min(height, y + h + pad), max(0, x - pad): min(width, x + w + pad)]
		h_roi, w_roi = roi.shape
		if h_roi == 0 or w_roi == 0:
			continue
		scale = 20.0 / max(h_roi, w_roi)
		new_w = max(1, int(w_roi * scale))
		new_h = max(1, int(h_roi * scale))
		resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
		canvas = np.zeros((28, 28), dtype=np.uint8)
		y_off = (28 - new_h) // 2
		x_off = (28 - new_w) // 2
		canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
		normalized = canvas.astype("float32") / 255.0
		aspect_ratio = w / float(h) if h > 0 else 0.0
		chars.append((normalized, aspect_ratio, (int(x), int(y), int(w), int(h))))
	return chars


def segment_region(image, bbox_cm, kind: str) -> List['Segment']:
	roi = crop_physical_roi(image, bbox_cm)
	binary, scaled = preprocess_roi_for_segmentation(roi)
	chars = segment_characters(binary, scaled)
	segments: List[Segment] = []
	for idx, (norm, aspect, bbox) in enumerate(sorted(chars, key=lambda c: c[2][0])):
		png_b64 = encode_png((norm * 255).astype("uint8"))
		segments.append(
			Segment(
				kind=kind,
				bbox=bbox,
				image_b64=f"data:image/png;base64,{png_b64}",
				aspect=aspect,
			)
		)
	return segments


def main():
	if len(sys.argv) < 2:
		print(json.dumps({"error": "usage: segment_digits.py <image_path> [mode]"}))
		return
	image_path = Path(sys.argv[1])
	mode = sys.argv[2] if len(sys.argv) > 2 else "full"
	img = cv2.imread(str(image_path))
	if img is None:
		print(json.dumps({"error": f"could not read image {image_path}"}))
		return

	segments: List[Segment] = []
	if mode in ("full", "dni"):
		segments.extend(segment_region(img, DNI_BOX_CM, kind="dni"))
	if mode == "full":
		segments.extend(segment_region(img, ANSWER_BOX_CM, kind="answer"))
	if mode == "raw":
		binary, scaled = preprocess_roi_for_segmentation(img)
		chars = segment_characters(binary, scaled)
		for norm, aspect, bbox in sorted(chars, key=lambda c: c[2][0]):
			png_b64 = encode_png((norm * 255).astype("uint8"))
			segments.append(
				Segment(
					kind="raw",
					bbox=bbox,
					image_b64=f"data:image/png;base64,{png_b64}",
					aspect=aspect,
				)
			)

	out = {
		"width": int(img.shape[1]),
		"height": int(img.shape[0]),
		"segments": [
			{
				"index": i,
				"bbox": seg.bbox,
				"aspect": seg.aspect,
				"image": seg.image_b64,
				"kind": seg.kind,
			}
			for i, seg in enumerate(segments)
		],
	}
	print(json.dumps(out))


if __name__ == "__main__":
	main()
