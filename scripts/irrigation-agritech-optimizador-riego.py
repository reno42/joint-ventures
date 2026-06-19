# ============================================================
# open-ai-toolkit | Optimizador de Riego Inteligente
# Rubro: AgriTech
# Para qué: Datos de suelo+clima → cuánto regar hoy
# Demo: datos sintéticos | Datos propios: CSV 3 meses (humedad%, temp, lluvia)
# ============================================================

!pip install scikit-learn pandas -q
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42); n = 500
df = pd.DataFrame({
    'humedad_suelo': np.random.uniform(10, 60, n),
    'temperatura': np.random.uniform(15, 38, n),
    'lluvia_ayer': np.random.exponential(5, n),
})
df['riego_optimo_L'] = (40 - 0.5*df['humedad_suelo'] + 0.8*df['temperatura'] - 2*df['lluvia_ayer'] + np.random.normal(0, 5, n)).clip(0, 100)

X_train, X_test, y_train, y_test = train_test_split(df.drop('riego_optimo_L', axis=1), df['riego_optimo_L'], test_size=0.2)
rf = RandomForestRegressor(100, max_depth=5).fit(X_train, y_train)

hoy = X_test.iloc[0:1]
print(f"💧 Riego recomendado hoy: {rf.predict(hoy)[0]:.1f} litros")
print(f"📊 Score R²: {rf.score(X_test, y_test):.2f}")
