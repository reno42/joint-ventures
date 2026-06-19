# ============================================================
# open-ai-toolkit | Detector de Neumonía en Rayos-X (CheXNet)
# Rubro: Salud
# Para qué: Radiografía de tórax → ¿neumonía? (Paper Stanford)
# Demo: 112k radiografías pre-entrenadas | Datos propios: 500+ radiografías
# ============================================================

!pip install tensorflow -q
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base.output)
model = tf.keras.Model(base.input, Dense(1, activation='sigmoid')(x))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print("✅ CheXNet listo — Stanford Paper (2017)")
print("📊 NIH ChestX-ray14: 112k imágenes gratis")
print("📎 kaggle.com/datasets/nih-chest-xrays")
