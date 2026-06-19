# ============================================================
# open-ai-toolkit | Analizador Geoespacial (PySal)
# Rubro: Ciencias Sociales / Políticas Públicas
# Para qué: Datos por distrito → clustering, segregación, patrones
# Demo: datos sintéticos | Datos propios: shapefile + indicadores
# ============================================================

!pip install pysal esda libpysal -q
import numpy as np
from esda.moran import Moran
from libpysal.weights import DistanceBand

np.random.seed(42); n = 30
coords = np.random.uniform(0, 10, (n, 2))
values = 50 + 20*np.sin(coords[:,0]) + np.random.normal(0, 5, n)

w = DistanceBand(coords, threshold=3.0)
moran = Moran(values, w)

print(f"📍 Moran's I: {moran.I:.3f} (p={moran.p_sim:.3f})")
print(f"   > 0 = agrupamiento | < 0 = dispersión | ≈ 0 = aleatorio")
print("🎯 Detecta patrones espaciales en crímenes, salud, educación")
