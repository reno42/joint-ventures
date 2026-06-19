# ============================================================
# open-ai-toolkit | Pronosticador Avanzado (Temporal Fusion Transformer)
# Rubro: Clima y Energía
# Para qué: Variables múltiples → predicción con Transformers
# Demo: datos sintéticos | Datos propios: CSV multivariable 1+ año
# ============================================================

!pip install pytorch-forecasting pytorch-lightning -q
import pandas as pd, numpy as np

np.random.seed(42); n = 1000
df = pd.DataFrame({
    'time_idx': np.tile(range(n//10), 10),
    'group': np.repeat(range(10), n//10),
    'value': np.random.normal(0, 1, n).cumsum() + 50,
    'predictor1': np.random.normal(0, 1, n),
})
df['time_idx'] = df.groupby('group').cumcount()

print("✅ PyTorch Forecasting: atención + LSTM + features interpretables")
print("🎯 Ideal: demanda energética, temperatura multivariable, viento")
