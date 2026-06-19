# ============================================================
# open-ai-toolkit | Detector de Plagas con YOLOv8
# Rubro: AgriTech
# Para qué: Subís foto de cultivo → detecta plagas/objetos
# Demo: 2 min | Datos propios: 50-100 fotos etiquetadas
# ============================================================

!pip install ultralytics -q
from ultralytics import YOLO
from google.colab import files
from PIL import Image

model = YOLO('yolov8n.pt')
print("✅ Modelo cargado (80 clases COCO)")

print("\n📸 Subí una imagen:")
uploaded = files.upload()
filename = list(uploaded.keys())[0]

results = model(filename)
for r in results:
    r.save(filename='resultado.jpg')
    display(Image.open('resultado.jpg'))

print("\n📊 DETECTADO:")
for r in results:
    for box in r.boxes:
        print(f"  🔍 {model.names[int(box.cls)]}: {float(box.conf):.0%}")
