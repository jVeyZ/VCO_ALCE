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
DEFAULT_QR_PATH = BASE_DIR / "img" / "qr_test"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def load_qr_metadata(image_path: Path) -> Optional[dict]:
    """Load QR metadata from JSON file in qr_test folder.
    Strips 'processed_' prefix from image stem if present to match original QR JSON files.
    """
    try:
        import json
        stem = image_path.stem
        # Remove 'processed_' prefix if present
        if stem.startswith("processed_"):
            stem = stem[len("processed_"):]
        json_path = DEFAULT_QR_PATH / f"{stem}.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load QR metadata for {image_path.stem}: {e}")
    return None

# Physical dimensions in centimeters for an A4 page.
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7

# Bounding box coordinates in centimeters, top-left origin.
BOUNDING_BOX_CM = (0.25, 8.45, 10.25, 9.90)  # (x1, y1, x2, y2)
START_ROW1 = (1.6, 11.6)
START_ROW2 =(11.6, 11.6)
SIZEOF_QUESTION = (9.5, 1.26)

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
    parser.add_argument(
        "-q", "--show-questions", action="store_true",
        help="Display segmented questions one by one."
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
        # Recreate the model architecture
        from tensorflow.keras import layers, models # type: ignore
        model = models.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(47, activation='softmax')
        ])
        # Load only the weights
        model.load_weights(str(model_path))
        return model
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
    batch = np.array(chars).reshape(-1, 28, 28, 1)
    preds = model.predict(batch, verbose=0)

    decoded = []
    for pred in preds:
        idx = int(np.argmax(pred))
        decoded.append(EMNIST_MAPPING.get(idx, '?'))

    return "".join(decoded)

def predict_multiple_choice_answer(question_box: np.ndarray) -> str:
    """Determine multiple choice answer (A, B, C, or D) by counting black pixels in each quarter.
    The question box is divided into 4 equal horizontal sections (A, B, C, D from left to right).
    Requires at least 40% of the section to be filled for a valid answer.
    """
    gray = cv2.cvtColor(question_box, cv2.COLOR_BGR2GRAY) if question_box.ndim == 3 else question_box
    
    # Threshold to get black pixels (marks)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Divide into 4 equal horizontal sections
    height, width = thresh.shape
    section_width = width // 4
    
    black_counts = []
    section_areas = []
    for i in range(4):
        x_start = i * section_width
        x_end = (i + 1) * section_width if i < 3 else width
        section = thresh[:, x_start:x_end]
        black_count = np.count_nonzero(section)
        section_area = section.size
        black_counts.append(black_count)
        section_areas.append(section_area)
    
    # Find the section with most black pixels
    max_idx = np.argmax(black_counts)
    answers = ['A', 'B', 'C', 'D']
    
    # Calculate percentage of section that is filled
    fill_percentage = (black_counts[max_idx] / section_areas[max_idx]) * 100 if section_areas[max_idx] > 0 else 0
    
    # Only return answer if at least 40% of the section is filled
    if fill_percentage >= 10:
        return answers[max_idx]
    return ""


def predict_numeric_answer(model, question_box: np.ndarray, show_debug=False) -> str:
    """Extract and recognize digits from a numeric answer question box."""
    binary, scaled_gray = preprocess_roi_for_segmentation(question_box)
    chars = segment_characters(binary, scaled_gray, show_debug=show_debug)
    
    if not chars:
        return ""
    
    # For numeric answers, all characters should be digits
    chars_data = [c[0] for c in chars]
    ratios = [c[1] for c in chars]
    
    batch = np.array(chars_data).reshape(-1, 28, 28, 1)
    preds = model.predict(batch, verbose=0)
    
    result = ""
    for i, pred in enumerate(preds):
        ratio = ratios[i]
        # Consider only digits 0-9 (indices 0-9)
        digit_score = pred[:10]
        idx = np.argmax(digit_score)
        char = EMNIST_MAPPING.get(idx, '?')
        
        # Heuristic: '1' vs '7'
        if char == '7' and ratio < 0.5:
            char = '1'
            
        result += char
    
    return result

def segment_questions(image, image_path: Path, show_debug=False) -> Tuple[List[np.ndarray], List[int]]:
    """Segment individual questions from the answer area.
    Uses total question count from QR metadata to determine segmentation.
    Row 1 (left column, vertical) has max 13 questions, Row 2 (right column, vertical) has remainder.
    Returns tuple of (question_boxes, numeric_question_indices).
    """
    img_height, img_width = image.shape[:2]
    question_width_cm, question_height_cm = SIZEOF_QUESTION

    # Convert centimeter sizes to pixels using image dimensions
    question_width_px = int((question_width_cm / A4_WIDTH_CM) * img_width)
    question_height_px = int((question_height_cm / A4_HEIGHT_CM) * img_height)

    # Load QR metadata to determine total questions and question types
    qr_metadata = load_qr_metadata(image_path)
    total_questions = 26  # Default to 26 (13 per column)
    numeric_indices = set()
    
    if qr_metadata and qr_metadata.get("results"):
        first_result = qr_metadata["results"][0]
        # Parse QR raw data: version;id;school;date;letter;num_questions;total_questions;numeric_indices
        if "raw" in first_result:
            try:
                raw_str = first_result["raw"]
                parts = raw_str.split(";")
                if len(parts) > 6:
                    total_questions = int(parts[6])
                if len(parts) > 7:
                    # Parse comma-separated numeric indices
                    numeric_str = parts[7].strip()
                    if numeric_str:
                        numeric_indices = set(int(x.strip()) for x in numeric_str.split(","))
            except (ValueError, IndexError, AttributeError):
                pass

    collected = []
    idx = 0
    
    # Determine how many questions go in each column (max 13 per column)
    col1_count = min(13, total_questions)
    col2_count = max(0, total_questions - 13)

    # COLUMN 1: Start at START_ROW1, go vertically top to bottom
    col1_x_start, col1_y_start = START_ROW1
    col1_x_start_px = int((col1_x_start / A4_WIDTH_CM) * img_width)
    col1_y_start_px = int((col1_y_start / A4_HEIGHT_CM) * img_height)
    
    for row in range(col1_count):
        x_start = col1_x_start_px
        y_start = col1_y_start_px + row * question_height_px
        x_end = x_start + question_width_px
        y_end = y_start + question_height_px
        
        # Clamp to image bounds
        x_start = max(0, min(x_start, img_width))
        x_end = max(0, min(x_end, img_width))
        y_start = max(0, min(y_start, img_height))
        y_end = max(0, min(y_end, img_height))
        
        if x_end > x_start and y_end > y_start:
            question_box = image[y_start:y_end, x_start:x_end]
            idx += 1
            q_type = "numeric" if idx in numeric_indices else "multiple-choice"
            print(f"[INFO] Question {idx} (col 1, row {row+1}): {q_type}")
            collected.append(question_box)
            if show_debug:
                disp = cv2.resize(question_box, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                cv2.imshow(f"Question {idx}", disp)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyWindow(f"Question {idx}")
                if key in (27, ord("q")):
                    break

    # COLUMN 2: Start at START_ROW2, go vertically top to bottom
    col2_x_start, col2_y_start = START_ROW2
    col2_x_start_px = int((col2_x_start / A4_WIDTH_CM) * img_width)
    col2_y_start_px = int((col2_y_start / A4_HEIGHT_CM) * img_height)
    
    for row in range(col2_count):
        x_start = col2_x_start_px
        y_start = col2_y_start_px + row * question_height_px
        x_end = x_start + question_width_px
        y_end = y_start + question_height_px
        
        # Clamp to image bounds
        x_start = max(0, min(x_start, img_width))
        x_end = max(0, min(x_end, img_width))
        y_start = max(0, min(y_start, img_height))
        y_end = max(0, min(y_end, img_height))
        
        if x_end > x_start and y_end > y_start:
            question_box = image[y_start:y_end, x_start:x_end]
            idx += 1
            q_type = "numeric" if idx in numeric_indices else "multiple-choice"
            print(f"[INFO] Question {idx} (col 2, row {row+1}): {q_type}")
            collected.append(question_box)
            if show_debug:
                disp = cv2.resize(question_box, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                cv2.imshow(f"Question {idx}", disp)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyWindow(f"Question {idx}")
                if key in (27, ord("q")):
                    break

    return collected, list(numeric_indices)

def process_image(image_path: Path, model, show_roi: bool, show_questions: bool = False) -> None:
    import json
    
    image = load_image(image_path)
    
    # Always segment and process questions
    question_boxes, numeric_indices = segment_questions(image, image_path, show_debug=show_questions)
    print(f"\n{image_path.name}: Segmented {len(question_boxes)} questions.")
    
    # Collect all answers
    answers = {}
    
    # Process all questions
    print(f"\nQuestion answers:")
    for idx, question_box in enumerate(question_boxes, start=1):
        if idx in numeric_indices:
            # Numeric question
            answer = predict_numeric_answer(model, question_box, show_debug=False)
            answers[str(idx)] = answer if answer else ""
            print(f"  Question {idx} (numeric): {answer if answer else '(no answer detected)'}")
        else:
            # Multiple choice question
            answer = predict_multiple_choice_answer(question_box)
            answers[str(idx)] = answer if answer else ""
            print(f"  Question {idx} (multiple choice): {answer if answer else '(no answer detected)'}")
    
    # Also process DNI
    roi = crop_physical_roi(image, BOUNDING_BOX_CM)
    
    if show_roi:
        cv2.imshow(f"roi::{image_path.name}", roi)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(f"roi::{image_path.name}")
        if key in (27, ord("q")):
            raise KeyboardInterrupt("ROI preview cancelled by user")
            
    binary, scaled_gray = preprocess_roi_for_segmentation(roi)
    chars = segment_characters(binary, scaled_gray, show_debug=show_roi)
    
    dni_code = ""
    if not chars:
        print(f"{image_path.name}: No DNI characters found.")
    else:
        # We expect 9 characters. If we found more or less, we might warn.
        if len(chars) != 9:
            print(f"{image_path.name}: [WARN] Found {len(chars)} characters, expected 9.")
        
        dni_code = predict_chars_constrained(model, chars)
        
        # Validate pattern
        match = re.match(r"^\d{8}[A-Z]$", dni_code)
        if match:
            print(f"{image_path.name}: DNI={dni_code}")
        else:
            print(f"{image_path.name}: DNI={dni_code} (Pattern mismatch)")
    
    # Extract QR metadata for exam information
    qr_data = load_qr_metadata(image_path)
    exam_info = {}
    
    if qr_data and qr_data.get("results"):
        try:
            first_result = qr_data["results"][0]
            if "raw" in first_result:
                raw_str = first_result["raw"]
                parts = raw_str.split(";")
                # Parse: version;id;school;date;letter;num_questions;total_questions;numeric_indices
                if len(parts) >= 5:
                    exam_info["version"] = parts[0]
                    exam_info["exam_id"] = parts[1]
                    exam_info["school"] = parts[2]
                    exam_info["date"] = parts[3]
                    exam_info["revision"] = parts[4]
                if len(parts) > 5:
                    exam_info["num_questions"] = int(parts[5])
                if len(parts) > 6:
                    exam_info["total_questions"] = int(parts[6])
        except (ValueError, IndexError, AttributeError):
            pass
    
    # Save results to JSON
    output_data = {
        "image": str(image_path),
        "name": image_path.name,
        "dni": dni_code,
        "exam": exam_info,
        "answers": answers
    }
    
    # Determine output filename (strip 'processed_' prefix if present)
    stem = image_path.stem
    if stem.startswith("processed_"):
        stem = stem[len("processed_"):]
    
    output_json_path = BASE_DIR / "img" / "json_test" / f"{stem}_answers.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_json_path}")


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
                process_image(image_path, model, args.show_roi, args.show_questions)
            except Exception as exc:
                print(f"[WARN] Failed to process {image_path}: {exc}")
    except KeyboardInterrupt:
        print("Aborted by user.")

if __name__ == "__main__":
    main()
