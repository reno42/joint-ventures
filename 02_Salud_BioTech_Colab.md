# 🏥 Salud / BioTech — 8 Herramientas en Colab

---

## 1. MONAI — IA para Imágenes Médicas ⭐

```python
# CELDA 1
!pip install monai nibabel -q
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss

# CELDA 2: Crear modelo
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
print(f"✅ MONAI UNet listo. Parámetros: {sum(p.numel() for p in model.parameters()):,}")
print("📊 Dataset: tcia.org (cáncer), decathlon (10 órganos), o tus DICOM")
print("🔗 Tutorial completo: github.com/Project-MONAI/tutorials")
```

**Tiempo:** 2 min | **GPU:** Recomendada | **Dataset:** DICOM/NIfTI de hospital

---

## 2. CheXNet — Detección de Neumonía en Rayos-X

```python
# CELDA 1
!pip install tensorflow -q
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# CELDA 2: Modelo DenseNet-121 pre-entrenado
base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base.output)
x = Dense(1, activation='sigmoid')(x)
model = Model(base.input, x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("✅ CheXNet listo")
print("📊 Dataset: NIH ChestX-ray14 (112k imágenes, gratuito)")
print("📎 kaggle.com/datasets/nih-chest-xrays")
```

**Tiempo:** 3 min | **GPU:** Recomendada | **Dataset:** NIH ChestX-ray14 (gratis, Kaggle)

---

## 3. nnU-Net — Segmentación Médica Auto-Configurable

```python
# CELDA 1
!pip install nnunetv2 -q

# CELDA 2
print("✅ nnU-Net instalado")
print("📦 nnU-Net se auto-configura: analiza tu dataset y elige la mejor arquitectura")
print("🔧 Uso básico:")
print("  1. Preparar datos en formato nnU-Net (nnUNetv2_plan_and_preprocess)")
print("  2. Entrenar (nnUNetv2_train)")
print("  3. Predecir (nnUNetv2_predict)")
print("\n📊 State-of-the-art en 33 benchmarks médicos internacionales")
print("📎 Tutorial: github.com/MIC-DKFZ/nnUNet")
```

**Tiempo:** 2 min (solo instalación) | **GPU:** Necesaria | **Dataset:** Imágenes + máscaras

---

## 4. BioBERT — NLP Biomédico

```python
# CELDA 1
!pip install transformers torch -q
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# CELDA 2: Cargar BioBERT
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('text-classification', model=model_name, tokenizer=tokenizer)

# CELDA 3: Probar
textos = [
    "The patient presented with acute myocardial infarction and elevated troponin levels.",
    "Aspirin 81mg daily was prescribed for cardiovascular prevention."
]
for t in textos:
    print(f"📝 {t[:80]}...")

print("\n✅ BioBERT cargado. Fine-tuneá con tus datos clínicos.")
```

**Tiempo:** 3 min (descarga modelo) | **GPU:** Recomendada | **Dataset:** PubMed textos

---

## 5. DeepLabCut — Tracking de Movimiento

```python
# CELDA 1
!pip install deeplabcut -q
import deeplabcut

# CELDA 2
print("✅ DeepLabCut instalado")
print("🎥 Flujo de trabajo:")
print("  1. Crear proyecto: deeplabcut.create_new_project()")
print("  2. Extraer frames de video")
print("  3. Etiquetar puntos clave (~20-50 frames)")
print("  4. Entrenar red neuronal")
print("  5. Analizar videos automáticamente")
print("\n🧠 Originalmente neurociencia. Ahora se usa en:")
print("  - Fisioterapia (marcha, postura)")
print("  - Deportes (técnica de movimiento)")
print("  - Veterinaria (análisis de comportamiento)")
```

**Tiempo:** 2 min | **GPU:** Recomendada | **Dataset:** Tus propios videos

---

## 6. AlphaFold (ColabFold) — Predicción de Estructura de Proteínas

```python
# CELDA 1
!pip install -q --no-warn-conflicts "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"

# CELDA 2
from colabfold.batch import get_queries, run
import os

# Secuencia de ejemplo: dominios de unión a ADN
queries = [
    ("test_protein", "MHHHHHHSSGVDLGTENLYFQSNAGSETVRFLAYDGWSFLASGGLGGQEAIAQAVGQALDA"),
]

# CELDA 3: Ejecutar AlphaFold (usa GPU T4 - gratis en Colab)
# Esto tarda ~5-10 min con GPU T4
try:
    results = run(
        queries=queries,
        num_models=1,
        use_amber=False,
    )
    print("✅ Estructura predicha. Revisá los archivos PDB generados.")
except Exception as e:
    print(f"⚠️ AlphaFold necesita ~12GB RAM GPU. Error: {e}")
    print("Si falló, usá Colab Pro o el notebook oficial:")
    print("📎 colab.research.google.com/github/sokrypton/ColabFold")
```

**Tiempo:** 5-10 min | **GPU:** T4 o mejor | **Dataset:** Secuencia de aminoácidos | **Premio Nobel 2024**

---

## 7. PathML — Patología Digital

```python
# CELDA 1
!pip install pathml -q
import pathml

print("✅ PathML instalado")
print("🔬 Análisis de Whole Slide Images (WSI) de patología")
print("📊 Flujo típico:")
print("  1. Cargar WSI (.svs, .tiff)")
print("  2. Preprocesar (normalizar color, tilear)")
print("  3. Aplicar modelo pre-entrenado (HoverNet para núcleos)")
print("  4. Cuantificar biomarcadores")
print("\n📎 Tutorial: github.com/Dana-Farber-AIOS/pathml")
```

**Tiempo:** 2 min | **GPU:** Recomendada | **Dataset:** WSI de biopsias

---

## 8. MedSAM — Segment Anything para Medicina

```python
# CELDA 1
!pip install git+https://github.com/bowang-lab/MedSAM.git -q
import torch

# CELDA 2
print("✅ MedSAM instalado")
print("🎯 Segment Anything Model fine-tuneado para imágenes médicas")
print("📸 Usalo así:")
print("  1. Cargá tu imagen médica")
print("  2. Dibujá un bounding box alrededor de lo que querés segmentar")
print("  3. MedSAM segmenta el órgano/tumor automáticamente")
print("\n📊 Entrenado en 1.5M+ máscaras de 11 modalidades médicas")
print("📎 Demo: github.com/bowang-lab/MedSAM")
```

**Tiempo:** 2 min | **GPU:** Necesaria | **Dataset:** Imagen + bounding box
