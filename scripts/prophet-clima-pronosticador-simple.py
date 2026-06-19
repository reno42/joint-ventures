# ============================================================
# open-ai-toolkit | Pronosticador Simple (Prophet)
# Rubro: Clima y Energía
# Para qué: CSV con fechas+valores → predicción 6 meses
# Demo: datos sintéticos | Datos propios: CSV (ds, y) 6+ meses
# ============================================================

!pip install prophet -q
import pandas as pd, numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=365*3, freq='D')
df = pd.DataFrame({'ds': dates, 'y': 20+10*np.sin(np.linspace(0,6*np.pi,365*3))+np.random.normal(0,2,365*3)})

m = Prophet(yearly_seasonality=True)
m.fit(df)
future = m.make_future_dataframe(periods=180)
forecast = m.predict(future)

m.plot(forecast)
plt.title("Pronóstico de Temperatura — Prophet")
plt.show()
print(f"📊 Predicción 180 días: {forecast['yhat'].iloc[-1]:.1f}°C")
