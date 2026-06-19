# ============================================================
# open-ai-toolkit | Predictor de Propiedades (MatGL)
# Rubro: Nanotecnología y Materiales
# Para qué: Estructura cristalina → energía, estabilidad, propiedades
# Demo: estructura Li2O de ejemplo | Datos propios: archivo CIF
# ============================================================

!pip install matgl pymatgen -q
import warnings
warnings.filterwarnings("ignore")

from matgl.models import M3GNet
from pymatgen.core import Structure, Lattice

# ── Cargar modelo ───────────────────────────────────────────
model = M3GNet.load()

# ── Crear estructura cristalina: Li2O ────────────────────────
lattice = Lattice.cubic(4.6)
structure = Structure(
    lattice,
    ["Li", "Li", "O"],
    [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]
)

# ── Predecir energía, fuerzas y stress ──────────────────────
pred = model.predict_structure(structure)

energy = float(pred['e'])
forces = pred['f']   # shape: (n_atoms, 3)
stress = pred['s']   # shape: (3, 3)

import numpy as np
max_force = np.max(np.abs(forces))

print(f"✅ M3GNet cargado | Material: {structure.composition.reduced_formula}")
print(f"⚡ Energía total: {energy:.4f} eV")
print(f"📐 Fuerza atómica máxima: {max_force:.4f} eV/Å")
print(f"📊 Stress tensor (GPa):")
for i, row in enumerate(stress):
    vals = "  ".join(f"{float(v):.2f}" for v in row)
    print(f"   [{vals}]")
print(f"🧪 {len(structure)} átomos — Predice energía, fuerzas y stress desde estructura cristalina")
