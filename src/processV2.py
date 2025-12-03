"""Extract ID code from a fixed physical region using a trained CNN."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

TF_ERROR = None
try:
    import tensorflow as tf  # type: ignore[import]
except ImportError as exc:
    tf = None
    TF_ERROR = exc

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "img" / "process_test"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "emnist_model.h5"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Physical dimensions in centimeters for an A4 page.
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7

# Bounding box coordinates in centimeters, top-left origin.
BOUNDING_BOX_CM = (0.25, 8.45, 10.25, 9.90)  # (x1, y1, x2, y2)

# EMNIST Mapping (0-9, A-Z)
EMNIST_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    # Mapping continues for lowercase if using Balanced/ByClass, but we focus on Uppercase for ID
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ID code using a trained CNN."
    )
    parser.add_argument(
        "-i", "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Directory with processed scans (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "-m", "--model-path", type=Path, default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained CNN model (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "-s", "--show-roi", action="store_true",
        help="Display the computed ROI and segmentation."
    )
    return parser.parse_args()

def ensure_tensorflow() -> None:
    if tf is None:
        raise ImportError(
            "tensorflow is required. Install it with 'pip install tensorflow'."
        ) from TF_ERROR

def load_cnn_model(model_path: Path):
    ensure_tensorflow()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        return tf.keras.models.load_model(str(model_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

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

def preprocess_roi_for_segmentation(roi):
    """Preprocess ROI to get a binary image suitable for contour detection."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Resize to make characters larger and easier to segment
    scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(scaled, (5, 5), 0)
    
    # Adaptive thresholding to handle lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to remove noise and connect broken parts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Closing fills small holes inside the foreground objects, or small black points on the object
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return cleaned, scaled

def segment_characters(binary_image, original_gray, show_debug=False) -> List[Tuple[np.ndarray, float]]:
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = binary_image.shape
    
    # Heuristics for character dimensions
    # Assuming the ROI contains mostly the ID code
    # Characters should be roughly same height
    
    char_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter based on size
        # Relaxed width threshold to catch thin characters like '1'
        if h > height * 0.2 and h < height * 0.9 and w > 2: # Min width 2 pixels
             char_candidates.append((x, y, w, h))
    
    # Sort by x coordinate (left to right)
    char_candidates.sort(key=lambda c: c[0])

    # Filter out noise based on vertical alignment
    if char_candidates:
        median_y = np.median([c[1] for c in char_candidates])
        # Keep only candidates that are roughly on the same line
        char_candidates = [c for c in char_candidates if abs(c[1] - median_y) < height * 0.3]

    # Handle merged characters (if we have fewer than 9)
    # Or if a character is unusually wide
    final_candidates = []
    if char_candidates:
        median_w = np.median([c[2] for c in char_candidates])
        for x, y, w, h in char_candidates:
            # If width is more than 1.8x median, it might be two characters merged
            if w > median_w * 1.8:
                # Split in half
                half_w = w // 2
                final_candidates.append((x, y, half_w, h))
                final_candidates.append((x + half_w, y, w - half_w, h))
            else:
                final_candidates.append((x, y, w, h))
        char_candidates = final_candidates

    # If we have more than 9, try to filter out the least likely ones (e.g. smallest area or furthest from center)
    if len(char_candidates) > 9:
        # Sort by area (descending) and take top 9? No, '1' has small area.
        # Sort by distance from center?
        # Let's assume the 9 characters are the most prominent ones.
        # Maybe filter by height consistency?
        median_h = np.median([c[3] for c in char_candidates])
        # Filter out those with height significantly different from median
        char_candidates = [c for c in char_candidates if abs(c[3] - median_h) < median_h * 0.3]
        
        # If still > 9, maybe sort by X and take the middle ones? 
        # Or just take the 9 with best aspect ratio?
        # For now, let's just take the first 9 if they are sorted by X, assuming noise is at the ends?
        # But noise could be in between.
        pass

    chars = []
    debug_img = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR) if show_debug else None
    
    for x, y, w, h in char_candidates:
        # Extract ROI
        # Add padding
        pad = 4
        roi = binary_image[max(0, y-pad):min(height, y+h+pad), max(0, x-pad):min(width, x+w+pad)]
        
        # Resize to 28x28 (EMNIST standard)
        # Maintain aspect ratio and pad
        h_roi, w_roi = roi.shape
        if h_roi == 0 or w_roi == 0: continue
        
        # Center of mass centering (like MNIST)
        # First, place in a 20x20 box preserving aspect ratio
        scale = 20.0 / max(h_roi, w_roi)
        new_h, new_w = int(h_roi * scale), int(w_roi * scale)
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create 28x28 canvas
        canvas = np.zeros((28, 28), dtype=np.uint8)
        
        # Center the 20x20 box in the 28x28 canvas based on center of mass
        # Calculate moments to find center of mass of the resized image
        M = cv2.moments(resized)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = new_w // 2, new_h // 2
            
        # Target center is (14, 14)
        # Shift to align center of mass to (14, 14)
        shift_x = 14 - cX
        shift_y = 14 - cY
        
        # Calculate top-left coordinates on canvas
        # We want the image to be placed such that its center of mass is at 14,14
        # But we also need to ensure it stays within bounds.
        # Actually, standard MNIST preprocessing centers the bounding box (20x20) in the 28x28 field by center of mass.
        # But simpler approach: Center the bounding box geometrically, then shift by CoM?
        # Let's stick to geometric centering for now, but maybe add a small border.
        # The previous code did geometric centering.
        
        # Let's try geometric centering first, but ensure the aspect ratio is handled well.
        y_off = (28 - new_h) // 2
        x_off = (28 - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        
        # Normalize to [0, 1] and add channel dimension
        normalized = canvas.astype("float32") / 255.0
        
        # Calculate aspect ratio of the original bounding box
        aspect_ratio = w / float(h) if h > 0 else 0
        chars.append((normalized, aspect_ratio))
        
        if show_debug:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    if show_debug:
        cv2.imshow("Segmentation", debug_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Segmentation")
        
    return chars

def predict_chars_constrained(model, chars_data: List[Tuple[np.ndarray, float]]) -> str:
    if not chars_data:
        return ""
    
    chars = [c[0] for c in chars_data]
    ratios = [c[1] for c in chars_data]
    
    # Batch prediction
    batch = np.array(chars).reshape(-1, 28, 28, 1)
    preds = model.predict(batch, verbose=0)
    
    result = ""
    for i, pred in enumerate(preds):
        ratio = ratios[i]
        if i < 8: # First 8 are digits
            # Consider only digits 0-9 (indices 0-9)
            digit_score = pred[:10]
            idx = np.argmax(digit_score)
            char = EMNIST_MAPPING.get(idx, '?')
            
            # Heuristic: '1' vs '7'
            # If predicted '7' but very thin, it's likely '1'
            # Threshold determined empirically: '7' ratio ~0.57, '1' ratio ~0.45
            if char == '7' and ratio < 0.5:
                char = '1'
                
            result += char
        elif i == 8: # 9th character is a letter
            # Consider only letters A-Z (indices 10-35)
            letter_score = pred[10:36]
            idx = np.argmax(letter_score) + 10
            result += EMNIST_MAPPING.get(idx, '?')
        else:
            # Extra characters? Just predict normally or ignore
            idx = np.argmax(pred)
            result += EMNIST_MAPPING.get(idx, '?')
            
    return result

def process_image(image_path: Path, model, show_roi: bool) -> None:
    image = load_image(image_path)
    roi = crop_physical_roi(image, BOUNDING_BOX_CM)
    
    if show_roi:
        cv2.imshow(f"roi::{image_path.name}", roi)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(f"roi::{image_path.name}")
        if key in (27, ord("q")):
            raise KeyboardInterrupt("ROI preview cancelled by user")
            
    binary, scaled_gray = preprocess_roi_for_segmentation(roi)
    chars = segment_characters(binary, scaled_gray, show_debug=show_roi)
    
    if not chars:
        print(f"{image_path.name}: No characters found.")
        return

    # We expect 9 characters. If we found more or less, we might warn.
    if len(chars) != 9:
        print(f"{image_path.name}: [WARN] Found {len(chars)} characters, expected 9.")
    
    code = predict_chars_constrained(model, chars)
    
    # Validate pattern
    match = re.match(r"^\d{8}[A-Z]$", code)
    if match:
        print(f"{image_path.name}: code={code}")
    else:
        print(f"{image_path.name}: code={code} (Pattern mismatch)")

def main() -> None:
    args = parse_args()
    
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
        
    images = list(gather_images(args.input_dir))
    if not images:
        raise FileNotFoundError(
            f"No images found in {args.input_dir}. Expected formats: {', '.join(sorted(SUPPORTED_EXTS))}"
        )
        
    print(f"Loading model from {args.model_path}...")
    try:
        model = load_cnn_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        for image_path in images:
            try:
                process_image(image_path, model, args.show_roi)
            except Exception as exc:
                print(f"[WARN] Failed to process {image_path}: {exc}")
    except KeyboardInterrupt:
        print("Aborted by user.")

if __name__ == "__main__":
    main()
