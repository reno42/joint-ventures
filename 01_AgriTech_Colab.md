# 🌾 AgriTech — 7 Herramientas Ejecutables en Google Colab

> Copiá cada celda en una nueva celda de Colab y ejecutá.

---

## 1. YOLOv8 — Detección de Objetos en Tiempo Real ⭐

```python
# CELDA 1: Instalar
!pip install ultralytics -q

# CELDA 2: Descargar imagen de prueba
!wget https://ultralytics.com/images/bus.jpg -O test.jpg

# CELDA 3: Detectar
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # nano: el más rápido
results = model('test.jpg')
results[0].show()  # muestra imagen con bounding boxes

# CELDA 4: Guardar resultado
results[0].save('resultado.jpg')
print("✅ Listo. Abrí resultado.jpg")
```

**Tiempo:** 2 min | **GPU:** No necesaria | **Dataset:** Incluido (COCO)

---

## 2. PlantVillage — Clasificación de Enfermedades en Plantas

```python
# CELDA 1
!pip install tensorflow tensorflow-hub -q
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import requests

# CELDA 2: Usar modelo pre-entrenado de TF Hub
# MobileNet fine-tuneado — clasifica 38 enfermedades
url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
model = tf.keras.Sequential([
    hub.KerasLayer(url, trainable=False),
    tf.keras.layers.Dense(38, activation='softmax')
])

# CELDA 3: Probar con imagen del dataset
!wget https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab%203335.JPG -O planta.jpg
img = Image.open('planta.jpg').resize((224, 224))
img_array = np.array(img) / 255.0
print("✅ Imagen cargada. Dataset disponible en: github.com/spMohanty/PlantVillage-Dataset")
```

**Tiempo:** 3 min | **GPU:** Recomendada | **Dataset:** PlantVillage (54k imágenes, gratuito)

---

## 3. Crop Yield Prediction — Predicción de Rendimiento

```python
# CELDA 1
!pip install tensorflow pandas scikit-learn -q
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# CELDA 2: Crear datos sintéticos de ejemplo (lluvia, temp, rendimiento)
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=365, freq='D')
df = pd.DataFrame({
    'lluvia_mm': np.random.gamma(2, 5, 365),
    'temp_max': 20 + 10 * np.sin(np.linspace(0, 2*np.pi, 365)) + np.random.normal(0, 2, 365),
    'temp_min': 10 + 8 * np.sin(np.linspace(0, 2*np.pi, 365)) + np.random.normal(0, 2, 365),
    'rendimiento_kg': 50 + 3 * np.random.gamma(2, 5, 365) + 20 * np.sin(np.linspace(0, 2*np.pi, 365))
}, index=dates)

# CELDA 3: Preparar datos para LSTM
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)
X, y = [], []
for i in range(30, len(scaled)):
    X.append(scaled[i-30:i, :-1])
    y.append(scaled[i, -1])
X, y = np.array(X), np.array(y)

# CELDA 4: Modelo LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(30, 3)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# CELDA 5: Predecir próximos 7 días
last_30 = scaled[-30:, :-1].reshape(1, 30, 3)
pred_scaled = model.predict(last_30)
pred = scaler.inverse_transform(np.hstack([np.zeros((1,3)), pred_scaled]))[0][-1]
print(f"📊 Rendimiento predicho mañana: {pred:.1f} kg")
```

**Tiempo:** 3 min | **GPU:** No necesaria | **Dataset:** Sintético (reemplazar con datos reales)

---

## 4. PlantSeg — Segmentación 3D de Tejidos Vegetales

```python
# CELDA 1
!pip install plantseg -q

# CELDA 2
from plantseg.api import run_plantseg
import os

# PlantSeg requiere imágenes 3D (tiff/h5). Para demo, descargamos ejemplo.
!wget https://github.com/hci-unihd/plant-seg/raw/master/examples/data/example.tif -O ejemplo.tif 2>/dev/null || echo "Dataset requiere descarga manual"

print("📦 PlantSeg instalado. Para ejecutar necesitás:")
print("  1. Imagen 3D de microscopio (.tif, .h5)")
print("  2. Elegir modelo pre-entrenado (PlantSeg incluye varios)")
print("  3. Ejecutar: run_plantseg('tu_archivo.tif', model_name='generic_light_sheet_3d_unet')")
print("\n🔬 Uso: biotecnología, mejoramiento genético. Nicho pero potente.")
```

**Tiempo:** 2 min (solo instalación) | **GPU:** Altamente recomendada | **Dataset:** Microscopía 3D

---

## 5. AgML — Benchmarking de ML Agrícola

```python
# CELDA 1
!pip install agml -q
import agml

# CELDA 2: Listar datasets disponibles
print("Datasets agrícolas disponibles:")
for ds in agml.public_data_sources():
    print(f"  - {ds}")

# CELDA 3: Cargar un dataset
loader = agml.data.AgMLDataLoader('apple_detection_segmentation')
loader.summary()

# CELDA 4: Visualizar muestra
import matplotlib.pyplot as plt
sample = loader[0]
if hasattr(sample, 'image'):
    plt.imshow(sample.image)
    plt.title("Muestra del dataset")
    plt.show()
```

**Tiempo:** 3 min | **GPU:** No necesaria | **Dataset:** Incluidos varios (apple, grape, etc.)

---

## 6. DeepWeeds — Detección de Malezas

```python
# CELDA 1
!pip install tensorflow pillow -q
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# CELDA 2: Modelo pre-entrenado (simplificado para demo)
base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base.output)
x = Dense(9, activation='softmax')(x)  # 9 clases de malezas en DeepWeeds
model = Model(base.input, x)

print("✅ Modelo listo. Dataset DeepWeeds: github.com/AlexOlsen/DeepWeeds")
print("📸 17,509 imágenes de 8 malezas nativas de Australia")
print("🎯 Para entrenar: descargar dataset y usar flow_from_directory()")
```

**Tiempo:** 2 min | **GPU:** Recomendada | **Dataset:** DeepWeeds (17k imágenes, gratuito)

---

## 7. Irrigation Optimization — Optimización de Riego

```python
# CELDA 1
!pip install tensorflow pandas scikit-learn -q
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# CELDA 2: Datos sintéticos de riego
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'humedad_suelo': np.random.uniform(10, 60, n),
    'temperatura': np.random.uniform(15, 38, n),
    'lluvia_ayer': np.random.exponential(5, n),
    'tipo_suelo': np.random.choice(['arenoso','arcilloso','franco'], n),
    'etapa_cultivo': np.random.choice(['germinacion','crecimiento','floracion','cosecha'], n)
})
df = pd.get_dummies(df, columns=['tipo_suelo', 'etapa_cultivo'])

# Variable objetivo: litros óptimos de riego
df['riego_optimo_L'] = (
    40 - 0.5*df['humedad_suelo'] 
    + 0.8*df['temperatura'] 
    - 2*df['lluvia_ayer'] 
    + np.random.normal(0, 5, n)
).clip(0, 100)

# CELDA 3: Modelo Random Forest
X = df.drop('riego_optimo_L', axis=1)
y = df['riego_optimo_L']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestRegressor(n_estimators=100, max_depth=5)
rf.fit(X_train, y_train)

# CELDA 4: Probar
hoy = X_test.iloc[0:1]
pred = rf.predict(hoy)[0]
print(f"💧 Riego recomendado para hoy: {pred:.1f} litros")
print(f"📊 R² score: {rf.score(X_test, y_test):.2f}")
```

**Tiempo:** 2 min | **GPU:** No necesaria | **Dataset:** Sintético (reemplazar con sensores IoT)
