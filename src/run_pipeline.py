import subprocess
import sys
from pathlib import Path

def run_command(script_name: str, args: list = None) -> int:
    cmd = [sys.executable, str(Path(__file__).parent / script_name)]
    if args:
        cmd.extend(args)
    
    result = subprocess.run(cmd)
    return result.returncode

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
    
    print("Completado!")

if __name__ == "__main__":
    main()
