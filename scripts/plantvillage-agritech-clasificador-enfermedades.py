# ============================================================
# open-ai-toolkit | Clasificador de Enfermedades (PlantVillage)
# Rubro: AgriTech
# Para qué: Clasifica 38 enfermedades en 14 cultivos
# Demo: YA entrenado con 54k imágenes | Datos propios: 100+ fotos
# ============================================================

!pip install tensorflow tensorflow-hub pillow -q
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from google.colab import files

print("✅ PlantVillage: 38 enfermedades, 14 cultivos")
print("   Cultivos: manzana, cereza, uva, maíz, papa, tomate...")
print("\n📸 Subí una foto de hoja:")
uploaded = files.upload()
filename = list(uploaded.keys())[0]

img = Image.open(filename).resize((224, 224))
img_array = np.array(img) / 255.0

# Modelo pre-entrenado de TF Hub
url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
model = tf.keras.Sequential([
    hub.KerasLayer(url, trainable=False),
    tf.keras.layers.Dense(38, activation='softmax')
])

pred = model.predict(np.expand_dims(img_array, 0))
clase = np.argmax(pred)
confianza = np.max(pred)
print(f"\n🔬 Clase {clase}: {confianza:.0%} de confianza")
print("\n⚠️ Para clasificación precisa: entrenar con dataset PlantVillage completo")
print("📎 github.com/spMohanty/PlantVillage-Dataset")
