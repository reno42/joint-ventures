# 🌍 Clima / Energía — 7 Herramientas en Colab

---

## 1. Prophet — Forecasting en 10 líneas ⭐

```python
# CELDA 1
!pip install prophet -q
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# CELDA 2: Datos sintéticos de temperatura
import numpy as np
dates = pd.date_range('2020-01-01', periods=365*3, freq='D')
df = pd.DataFrame({
    'ds': dates,
    'y': 20 + 10 * np.sin(np.linspace(0, 6*np.pi, 365*3)) + np.random.normal(0, 2, 365*3)
})

# CELDA 3: Entrenar y predecir
m = Prophet(yearly_seasonality=True, daily_seasonality=False)
m.fit(df)
future = m.make_future_dataframe(periods=180)
forecast = m.predict(future)

# CELDA 4: Visualizar
m.plot(forecast)
plt.title("Pronóstico de Temperatura — Prophet")
plt.show()

print(f"📊 Temperatura predicha en 180 días: {forecast['yhat'].iloc[-1]:.1f}°C")
```

**Tiempo:** 1 min | **GPU:** NO | **Dataset:** CSV con fechas y valores

---

## 2. PyTorch Forecasting — Temporal Fusion Transformer

```python
# CELDA 1
!pip install pytorch-forecasting pytorch-lightning -q
import torch
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

# CELDA 2: Datos de ejemplo
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'time_idx': np.tile(range(n//10), 10),
    'group': np.repeat(range(10), n//10),
    'value': np.random.normal(0, 1, n).cumsum() + 50,
    'predictor1': np.random.normal(0, 1, n),
    'month': np.tile(np.tile(range(12), n//120+1)[:n//10], 10)
})
df['time_idx'] = df.groupby('group').cumcount()

print("✅ PyTorch Forecasting listo")
print("📊 Temporal Fusion Transformer: atención + LSTM + features interpretables")
print("🎯 Ideal para: demanda energética, temperatura multivariable, viento")
```

**Tiempo:** 2 min | **GPU:** Opcional | **Dataset:** Series temporales multivariables

---

## 3. OpenClimateFix — Nowcasting Solar

```python
# CELDA 1
!pip install ocf_datapipes -q

# CELDA 2
print("✅ OpenClimateFix instalado")
print("☀️ Nowcasting de energía solar: predecir producción próximas 4 horas")
print("📡 Usa imágenes de satélite + datos de parques solares")
print("🏭 Industria: operadores de red eléctrica necesitan saber cuánta solar entra")
print("\n📊 UK tiene 100+ parques solares usando esto en producción")
print("📎 github.com/openclimatefix")
```

**Tiempo:** 2 min | **GPU:** Recomendada | **Dataset:** Satélite + producción solar

---

## 4. ClimateLearn — Forecasting Climático

```python
# CELDA 1
!pip install climate-learn -q
import climate_learn as cl

# CELDA 2
print("✅ ClimateLearn instalado")
print("🌡️ Forecasting + downscaling climático con Deep Learning")
print("📊 Datos que usa:")
print("   - ERA5 (reanalysis global, ~30km resolución, gratis)")
print("   - CMIP6 (modelos climáticos futuros)")
print("   - Estaciones meteorológicas locales")
print("\n🎯 Ideal para: bajar resolución de 30km a 1km para predicción regional")
```

**Tiempo:** 2 min | **GPU:** Recomendada | **Dataset:** ERA5 (gratis, CDS Copernicus)

---

## 5. FourCastNet — Predicción Climática Global (NVIDIA)

```python
# CELDA 1
!pip install modulus -q

# CELDA 2
print("✅ NVIDIA Modulus instalado (incluye FourCastNet)")
print("🌍 Predicción climática global con Fourier Neural Operator")
print("⚡ 45,000x más rápido que modelos físicos tradicionales")
print("📊 Predice: viento, temperatura, presión, humedad a nivel global")
print("\n🔗 Modelo de NVIDIA. Paper: arxiv.org/abs/2202.11214")
print("⚠️ Requiere dataset ERA5 completo (~50GB). Para testing usá pesos pre-entrenados.")
```

**Tiempo:** 2 min | **GPU:** A100 recomendada | **Dataset:** ERA5 (50GB)

---

## 6. DeepSensor — Interpolación Espacial

```python
# CELDA 1
!pip install deepsensor -q
import deepsensor as ds
import numpy as np
import pandas as pd

# CELDA 2: Datos espaciales de ejemplo
np.random.seed(42)
n_stations = 20
lats = np.random.uniform(-34, -32, n_stations)
lons = np.random.uniform(-58, -56, n_stations)
temps = 25 + 5 * np.sin(lats * 10) + np.random.normal(0, 2, n_stations)

df = pd.DataFrame({'lat': lats, 'lon': lons, 'temp': temps})
print("✅ DeepSensor listo")
print(f"📍 {n_stations} estaciones meteorológicas virtuales")
print("🎯 Predice temperatura en ubicaciones sin estación usando interpolación DL")
```

**Tiempo:** 2 min | **GPU:** Opcional | **Dataset:** Coordenadas + mediciones

---

## 7. Wind Power Forecasting

```python
# CELDA 1
!pip install tensorflow pandas scikit-learn -q
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# CELDA 2: Datos sintéticos de viento
np.random.seed(42)
hours = 24 * 90
wind = 15 + 8 * np.sin(np.linspace(0, 8*np.pi, hours)) + np.random.normal(0, 3, hours)
power = np.maximum(0, (wind - 4) * 20 + np.random.normal(0, 10, hours))

# CELDA 3: Preparar para LSTM
scaler = MinMaxScaler()
scaled = scaler.fit_transform(np.column_stack([wind, power]))
X, y = [], []
for i in range(72, len(scaled)):
    X.append(scaled[i-72:i, 0])
    y.append(scaled[i, 1])
X, y = np.array(X).reshape(-1, 72, 1), np.array(y)

# CELDA 4: Modelo LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(72, 1)),
    Dropout(0.2),
    LSTM(50),
    Dense(1)
])
model.compile('adam', 'mse')
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# CELDA 5: Predecir
last_wind = scaled[-72:, 0].reshape(1, 72, 1)
pred = model.predict(last_wind)[0][0]
print(f"💨 Potencia eólica predicha próx. hora: {pred:.1f} (normalizada)")
```

**Tiempo:** 2 min | **GPU:** NO | **Dataset:** Velocidad viento + producción
