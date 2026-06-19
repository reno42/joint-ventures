# 🤖 Open AI Toolkit — Detección de Objetos con YOLOv8
# Repo: open-ai-toolkit
# Descripción: Herramientas de IA open-source ejecutables en Google Colab.
#             Diseñado para investigadores, emprendedores y científicos en LATAM.
#             Sin instalación. Sin costo. Sin GPU propia.
# Autor: Renzo Perez Bartra — Joint Venture / Futuraria
# Fecha: 2026-05-27

# ============================================================
# CELDA 1 — Instalar dependencias
# ============================================================
!pip install ultralytics -q
from ultralytics import YOLO
from google.colab import files
from PIL import Image
import os

print("✅ YOLOv8 instalado. Listo para detectar objetos.")

# ============================================================
# CELDA 2 — Descargar modelo pre-entrenado
# ============================================================
model = YOLO('yolov8n.pt')  # nano: el más rápido. También: yolov8s.pt, yolov8m.pt, yolov8l.pt
print("✅ Modelo YOLOv8n cargado (80 clases COCO)")
print("   Clases incluidas: persona, auto, perro, gato, avión, botella, celular, laptop...")

# ============================================================
# CELDA 3 — Subir TU imagen
# ============================================================
print("📸 Seleccioná una imagen de tu computadora:")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(f"✅ Imagen cargada: {filename}")

# ============================================================
# CELDA 4 — Detectar objetos en tu imagen
# ============================================================
results = model(filename)

# Mostrar resultado en la notebook
from IPython.display import display
for r in results:
    r.save(filename='resultado.jpg')
    display(Image.open('resultado.jpg'))

# ============================================================
# CELDA 5 — Mostrar qué encontró (consola)
# ============================================================
print("\n📊 OBJETOS DETECTADOS:")
for r in results:
    boxes = r.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            clase = model.names[int(box.cls)]
            confianza = float(box.conf)
            print(f"  🔍 {clase}: {confianza:.2%} de confianza")
    else:
        print("  ❌ No se detectaron objetos en esta imagen.")

# ============================================================
# CELDA 6 (OPCIONAL) — Detectar en tiempo real con tu webcam
# ============================================================
# Descomentá las siguientes líneas si querés probar con cámara:
# from IPython.display import display, Javascript
# from google.colab.output import eval_js
# print("📹 Para webcam ejecutá esto en una nueva celda y permití acceso a la cámara")
