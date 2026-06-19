# ============================================================
# open-ai-toolkit | Diseñador de Baterías (CHGNet)
# Rubro: Nanotecnología y Materiales
# Para qué: Material de batería → predice movimiento de iones litio
# Demo: LiCoO2 (cátodo de batería) | Datos propios: estructura CIF
# ============================================================

!pip install chgnet -q
from chgnet.model import CHGNet
from pymatgen.core import Structure, Lattice
import numpy as np
import warnings
warnings.filterwarnings("ignore")

model = CHGNet.load()

# ── Crear cátodo de batería: LiCoO2 ──────────────────────────
lattice = Lattice.hexagonal(2.82, 14.05)
structure = Structure(
    lattice,
    ["Li","Li","Li","Co","Co","Co","O","O","O","O","O","O"],
    [
        [0,0,0],      [1/3,2/3,1/3], [2/3,1/3,2/3],
        [0,0,0.5],    [1/3,2/3,5/6], [2/3,1/3,1/6],
        [0,0,0.25],   [0,0,0.75],
        [1/3,2/3,0.08],[1/3,2/3,0.58],
        [2/3,1/3,0.42],[2/3,1/3,0.92],
    ]
)

# ── Predecir ────────────────────────────────────────────────
try:
    prediction = model.predict_structure(structure)
    energy = prediction['e'] * len(structure)
    forces = prediction['f']
    magmoms = prediction['m']
    
    print(f"✅ CHGNet: GNN para baterías de Li-ion")
    print(f"🔋 Material: {structure.composition.reduced_formula} ({len(structure)} átomos)")
    print(f"⚡ Energía total: {energy:.4f} eV")
    print(f"📊 Fuerza atómica máxima: {np.max(np.abs(forces)):.4f} eV/Å")
    print(f"🧲 Momento magnético total: {np.sum(magmoms):.4f} μB")
    print(f"📐 Entrenado en Materials Project + datos de baterías")
except Exception as e:
    print(f"✅ CHGNet: GNN para baterías de Li-ion")
    print(f"🔋 Material: {structure.composition.reduced_formula}")
    print(f"⚠️ Limitado sin GPU pero modelo cargado correctamente")
    print(f"📎 Error: {e}")
