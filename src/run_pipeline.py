import subprocess
import sys
from pathlib import Path

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
        base_dir / "img" / "json_test"
    ]
    
    print("\n[CLEANUP] Eliminando directorios de prueba...")
    for test_dir in test_dirs:
        if test_dir.exists():
            print(f"  Removing: {test_dir}")
            shutil.rmtree(test_dir)
    
    print("[CLEANUP] Done!")

def main() -> None:
    print("\n[STEP 1/3] Preprocesando imágenes...")
    if run_command("preprocess.py") != 0:
        sys.exit(1)
    
    print("\n[STEP 2/3] Extrayendo códigos QR...")
    if run_command("qr_reader_cpp.py") != 0:
        sys.exit(1)
    
    print("\n[STEP 3/3] Procesando imágenes (DNI y respuestas)...")
    if run_command("process.py") != 0:
        sys.exit(1)

    cleanup_directories()
    
    print("Completado!")

if __name__ == "__main__":
    main()
