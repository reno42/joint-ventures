# 🌾 Cómo entrenar YOLOv8 con fotos de cultivos (en español)

> Para: investigadores y agrónomos en Argentina
> Repo: open-ai-toolkit

---

## 📸 Opción A: Entrenar CON TUS PROPIAS FOTOS

Si tenés fotos de cultivos con plagas/enfermedades etiquetadas.

```python
# CELDA 1 — Instalar
!pip install ultralytics -q
from ultralytics import YOLO
from google.colab import files
import os
import yaml

# ============================================================
# CELDA 2 — Subí tus fotos etiquetadas
# ============================================================
# Necesitás una carpeta así en tu PC:
#   mi_cultivo/
#     images/train/   ← las fotos
#     images/val/     ← fotos para validar
#     labels/train/   ← archivos .txt con las coordenadas de cada plaga

!mkdir -p dataset/images/train dataset/images/val dataset/labels/train dataset/labels/val
print("📁 Subí tus imágenes a dataset/images/train/")
print("📁 Subí tus etiquetas a dataset/labels/train/")

# Subir imágenes
print("\n📸 Subí las FOTOS (jpg, png):")
uploaded = files.upload()
for filename in uploaded.keys():
    os.rename(filename, f'dataset/images/train/{filename}')

# Subir etiquetas
print("\n📝 Subí las ETIQUETAS (txt):")
uploaded = files.upload()
for filename in uploaded.keys():
    os.rename(filename, f'dataset/labels/train/{filename}')

# ============================================================
# CELDA 3 — Configurar el dataset
# ============================================================
# Crear archivo data.yaml
data_yaml = {
    'path': '/content/dataset',
    'train': 'images/train',
    'val': 'images/val',
    'names': {
        0: 'plaga',
        1: 'enfermedad',
        2: 'maleza'
    },
    'nc': 3  # número de clases
}

with open('data.yaml', 'w') as f:
    yaml.dump(data_yaml, f)

print("✅ Dataset configurado")
print(f"📊 Clases: {list(data_yaml['names'].values())}")

# ============================================================
# CELDA 4 — Entrenar el modelo
# ============================================================
model = YOLO('yolov8n.pt')  # empezamos del modelo pre-entrenado

results = model.train(
    data='data.yaml',
    epochs=50,        # más épocas = más aprendizaje (pero más tiempo)
    imgsz=640,        # tamaño de imagen
    patience=10,      # para si no mejora en 10 épocas seguidas
    batch=16,
    name='mi_cultivo'
)

print("✅ Entrenamiento completado!")

# ============================================================
# CELDA 5 — Probar con una foto nueva
# ============================================================
model = YOLO('runs/detect/mi_cultivo/weights/best.pt')

print("📸 Subí una foto NUEVA para probar:")
uploaded = files.upload()
test_img = list(uploaded.keys())[0]

results = model(test_img)
results[0].show()
```

---

## 📦 Opción B: Usar el dataset PlantVillage (YA ETIQUETADO)

Si no tenés fotos propias, usá este dataset gratuito con 54,000 imágenes de 38 enfermedades en 14 cultivos.

```python
# CELDA 1
!pip install ultralytics roboflow -q
from ultralytics import YOLO

# CELDA 2 — Descargar PlantVillage desde Roboflow (gratis)
# Dataset: 38 enfermedades de plantas, 54k imágenes, YA ETIQUETADO
!mkdir -p /content/plantvillage
%cd /content/plantvillage

# Opción: descargar manualmente de Roboflow
# 1. Andá a https://universe.roboflow.com/
# 2. Buscá "PlantVillage" → elegí un dataset
# 3. Copiá el código de descarga y pegalo acá

from roboflow import Roboflow
rf = Roboflow(api_key="TU_API_KEY_GRATIS")  # ← registrate gratis en roboflow.com
project = rf.workspace("plantvillage").project("plant-village-detection")
dataset = project.version(1).download("yolov8")

# CELDA 3 — Entrenar con PlantVillage
model = YOLO('yolov8n.pt')
results = model.train(
    data=f'{dataset.location}/data.yaml',
    epochs=30,
    imgsz=640,
    name='plantvillage'
)

# CELDA 4 — Probar
model = YOLO('runs/detect/plantvillage/weights/best.pt')
from google.colab import files
uploaded = files.upload()
test_img = list(uploaded.keys())[0]
results = model(test_img)
results[0].show()
```

---

## ⏱️ Tiempos

| Dataset | Imágenes | Tiempo Colab Free |
|---------|:--------:|:-----------------:|
| 50-100 propias | 50-100 | ~20 min |
| PlantVillage completo | 54,000 | ~2 horas |
| 500 propias | 500 | ~40 min |

---

## 🎯 Las 7 herramientas de AgriTech (para qué sirve cada una)

| # | Herramienta | ¿Entrena? | ¿Para qué? |
|---|------------|:---------:|-----------|
| 1 | **YOLOv8** | ✅ SÍ | Detectar plagas visualmente (fotos) |
| 2 | PlantVillage | 📦 Dataset | Dataset YA etiquetado de enfermedades |
| 3 | Predicción Rendimiento | ✅ SÍ | Predecir cosecha con datos de clima |
| 4 | PlantSeg | ✅ SÍ | Microscopía 3D (laboratorio) |
| 5 | AgML | 📊 Comparar | Probar varios modelos y elegir el mejor |
| 6 | DeepWeeds | 📦 Dataset | Dataset de malezas (Australia) |
| 7 | Riego | ✅ SÍ | Optimizar cuándo y cuánto regar |

**Para tu caso:** YOLOv8 solo alcanza. Las otras son para tareas diferentes.
