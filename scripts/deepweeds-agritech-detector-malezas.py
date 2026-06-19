# ============================================================
# open-ai-toolkit | Detector de Malezas (DeepWeeds)
# Rubro: AgriTech
# Para qué: Clasifica 8 tipos de malezas con ResNet
# Demo: 17,509 imágenes | Datos propios: 50-100 fotos de malezas locales
# ============================================================

!pip install tensorflow pillow -q
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base.output)
model = tf.keras.Model(base.input, Dense(9, activation='softmax')(x))

print("✅ DeepWeeds: 8 malezas nativas de Australia")
print("📸 17,509 imágenes etiquetadas")
print("🎯 Para malezas argentinas: entrenar con fotos locales")
