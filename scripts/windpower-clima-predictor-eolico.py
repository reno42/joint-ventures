# ============================================================
# open-ai-toolkit | Predictor de Energía Eólica
# Rubro: Clima y Energía
# Para qué: Viento → producción eólica próximas 72h
# Demo: datos sintéticos | Datos propios: CSV viento+producción 3 meses
# ============================================================

!pip install tensorflow pandas scikit-learn -q
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

np.random.seed(42)
hours = 24*90
wind = 15+8*np.sin(np.linspace(0,8*np.pi,hours))+np.random.normal(0,3,hours)
power = np.maximum(0,(wind-4)*20+np.random.normal(0,10,hours))

scaler = MinMaxScaler()
scaled = scaler.fit_transform(np.column_stack([wind,power]))
X, y = [], []
for i in range(72, len(scaled)):
    X.append(scaled[i-72:i,0]); y.append(scaled[i,1])
X, y = np.array(X).reshape(-1,72,1), np.array(y)

model = Sequential([LSTM(50,return_sequences=True,input_shape=(72,1)),Dropout(0.2),LSTM(50),Dense(1)])
model.compile('adam','mse'); model.fit(X,y,epochs=5,batch_size=32,verbose=1)

pred = model.predict(scaled[-72:,0].reshape(1,72,1))[0][0]
print(f"💨 Potencia eólica predicha próx. hora: {pred:.3f} (normalizada)")
