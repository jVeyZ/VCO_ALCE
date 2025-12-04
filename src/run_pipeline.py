import subprocess
import sys
import shutil
from pathlib import Path

def copy_input_files() -> None:
    """Copy files from input folder to img/preprocess_test."""
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "input"
    preprocess_dir = base_dir / "img" / "preprocess_test"
    
    if not input_dir.exists():
        print(f"[WARNING] Input directory not found: {input_dir}")
        return
    
    print(f"\n[COPY] Copying files from {input_dir} to {preprocess_dir}...")
    
    # Create preprocess_test directory if it doesn't exist
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all files from input to preprocess_test
    image_files = list(input_dir.glob("*.*"))
    if not image_files:
        print("[WARNING] No files found in input directory")
        return
    
    for file_path in image_files:
        if file_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}:
            dest_path = preprocess_dir / file_path.name
            print(f"  Copying: {file_path.name}")
            shutil.copy2(file_path, dest_path)
    
    print(f"[COPY] Done! Copied {len(image_files)} files.")

def run_command(script_name: str, args: list = None) -> int:
    cmd = [sys.executable, str(Path(__file__).parent / script_name)]
    if args:
        cmd.extend(args)
    
    result = subprocess.run(cmd)
    return result.returncode

def cleanup_directories() -> None:
    base_dir = Path(__file__).parent.parent
    test_dirs = [
        base_dir / "img" / "process_test",
        base_dir / "img" / "qr_test",
        base_dir / "img" / "preprocess_test",
        base_dir / "img" / "json_test",
    ]
    
    print("\n[CLEANUP] Eliminando directorios de prueba...")
    for test_dir in test_dirs:
        if test_dir.exists():
            print(f"  Removing: {test_dir}")
            shutil.rmtree(test_dir)
    
    print("[CLEANUP] Done!")

def move_json_to_output() -> None:
    """Move json_test folder to output folder in root."""
    base_dir = Path(__file__).parent.parent
    json_test_dir = base_dir / "img" / "json_test"
    src_json_test = Path(__file__).parent / "json_test"
    output_dir = base_dir / "output"
    output_json_dir = output_dir / "json_test"
    
    # First, check if src/json_test exists (from process.py output)
    source_dir = src_json_test if src_json_test.exists() else json_test_dir
    
    if not source_dir.exists():
        print("[WARNING] json_test directory not found")
        return
    
    print(f"\n[MOVE] Moving {source_dir} to {output_json_dir}...")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove output/json_test if it already exists
    if output_json_dir.exists():
        print(f"  Removing existing: {output_json_dir}")
        shutil.rmtree(output_json_dir)
    
    # Move json_test to output
    shutil.move(str(source_dir), str(output_json_dir))
    print(f"[MOVE] Done!")

def main() -> None:
    print("\n[STEP 0/4] Copying input files...")
    copy_input_files()
    
    print("\n[STEP 1/4] Preprocesando imágenes...")
    if run_command("preprocess.py") != 0:
        sys.exit(1)
    
    print("\n[STEP 2/4] Extrayendo códigos QR...")
    if run_command("qr_reader_cpp.py") != 0:
        sys.exit(1)
    
    print("\n[STEP 3/4] Procesando imágenes (DNI y respuestas)...")
    if run_command("process.py") != 0:
        sys.exit(1)

    print("\n[STEP 4/5] Finalizing...")
    
    print("\n[STEP 5/5] Corrigiendo respuestas...")
    if run_command("correct.py") != 0:
        print("[WARNING] Correction step failed, but continuing...")
    
    cleanup_directories()
    
    print("\n✓ ¡Completado!")

if __name__ == "__main__":
    main()

