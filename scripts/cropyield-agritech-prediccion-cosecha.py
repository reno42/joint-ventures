# ============================================================
# open-ai-toolkit | Predicción de Rendimiento de Cosecha
# Rubro: AgriTech
# Para qué: Datos de clima → predice kg/ha de la próxima temporada
# Demo: datos sintéticos | Datos propios: 1 año CSV (fecha, lluvia, temp, rendimiento)
# ============================================================

!pip install tensorflow pandas scikit-learn -q
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Datos sintéticos
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=365, freq='D')
df = pd.DataFrame({
    'lluvia_mm': np.random.gamma(2, 5, 365),
    'temp_max': 20 + 10 * np.sin(np.linspace(0, 2*np.pi, 365)) + np.random.normal(0, 2, 365),
    'temp_min': 10 + 8 * np.sin(np.linspace(0, 2*np.pi, 365)) + np.random.normal(0, 2, 365),
    'rendimiento_kg': 50 + 3 * np.random.gamma(2, 5, 365) + 20 * np.sin(np.linspace(0, 2*np.pi, 365))
}, index=dates)

scaler = MinMaxScaler(); scaled = scaler.fit_transform(df)
X, y = [], []
for i in range(30, len(scaled)):
    X.append(scaled[i-30:i, :-1]); y.append(scaled[i, -1])
X, y = np.array(X), np.array(y)

model = Sequential([LSTM(50, return_sequences=True, input_shape=(30,3)), LSTM(50), Dense(1)])
model.compile('adam', 'mse'); model.fit(X, y, epochs=5, verbose=1)

last = scaled[-30:, :-1].reshape(1, 30, 3)
pred = scaler.inverse_transform(np.hstack([np.zeros((1,3)), model.predict(last)]))[0][-1]
print(f"\n🌾 Rendimiento predicho próxima temporada: {pred:.1f} kg/ha")
