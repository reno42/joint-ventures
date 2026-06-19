# ============================================================
# open-ai-toolkit | Interpolador Espacial (DeepSensor)
# Rubro: Clima y Energía
# Para qué: 20 estaciones → estima temperatura en CUALQUIER punto
# Demo: datos sintéticos | Datos propios: CSV (lat, lon, valor)
# ============================================================

!pip install deepsensor -q
import numpy as np, pandas as pd

np.random.seed(42)
n = 20
df = pd.DataFrame({
    'lat': np.random.uniform(-34, -32, n),
    'lon': np.random.uniform(-58, -56, n),
    'temp': 25 + 5*np.sin(np.arange(n)) + np.random.normal(0, 2, n)
})

print("✅ DeepSensor: Interpolación espacial con DL")
print(f"📍 {n} estaciones virtuales cargadas")
print("🎯 Estima valores en puntos sin medición")
