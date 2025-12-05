### EJECUTAR RECONOCIMIENTO
1. Poner fotos en img/preprocess_test
2. Ejecutar preprocess_test.py
3. Ejecutar qr_reader_cpp.py
4. Ejecutar process.py -q (para ver las preguntas segmentadas)

## Entrenar el modelo EMNIST

1. Crear y activar un entorno virtual, luego instalar dependencias:
	```bash
	python -m venv .venv
	source .venv/bin/activate  # Windows: .venv\Scripts\activate
	pip install --upgrade pip
	pip install -r requirements.txt
	```
2. (Opcional) Definir `TFDS_DATA_DIR=/ruta/a/data/EMNIST` si ya descargaste el dataset de forma manual.
3. Lanzar el nuevo entrenamiento de alta precisión (mezcla de datos, regularización fuerte y scheduler cíclico):
	```bash
	python src/train_model.py
	```
	El script aplica aumentos avanzados (rotaciones, zoom, contraste, ruido y mixup), regularización (dropout espacial + L2) y callbacks (CosineDecay + EarlyStopping + ReduceLROnPlateau + ModelCheckpoint) para maximizar la `val_accuracy`. El mejor modelo se guarda automáticamente en `models/emnist_model.h5` y se evalúa al final sobre el split de test.