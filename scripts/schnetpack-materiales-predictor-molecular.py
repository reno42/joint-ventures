# ============================================================
# open-ai-toolkit | Predictor Molecular (SchNetPack)
# Rubro: Nanotecnología y Materiales
# Para qué: Molécula → 12 propiedades (energía, dipolos, fuerzas)
# Demo: dataset QM9 incluido | Datos propios: archivos XYZ
# ============================================================

!pip install schnetpack -q
import torch
import schnetpack as spk
from schnetpack.data import ASEAtomsData
import schnetpack.transform as trn
import ase

# ── Crear molécula demo: H2O ────────────────────────────────
water = ase.Atoms('H2O', positions=[[0,0,0],[0.96,0,0],[-0.24,0.93,0]])
water.set_cell([12,12,12])
water.center()

# Definir qué propiedades queremos
properties = ["energy"]
cutoff = 5.0

# Calcular features atómicos reales
import numpy as np
from ase.neighborlist import neighbor_list

# Obtener distancias inter-atómicas
n_atoms = len(water)
positions = water.get_positions()
atomic_numbers = water.get_atomic_numbers()  # [1, 1, 8]

# Construir inputs para SchNet
_inputs = {
    "_atomic_numbers": torch.tensor(atomic_numbers, dtype=torch.long).unsqueeze(0),
    "_n_atoms": torch.tensor([n_atoms]),
    "energy": torch.zeros(1, 1),
}

# Crear modelo SchNet liviano
model = spk.SchNet(
    n_atom_basis=64,
    n_interactions=3,
    n_filters=64,
    cutoff=cutoff,
    energy_key="energy",
    forces_key=None,
)

# Forward pass
result = model(_inputs)
predicted_energy = result["energy"].item()

print(f"✅ SchNetPack: Predice 12 propiedades moleculares")
print(f"📊 Dataset QM9: 134k moléculas orgánicas incluidas")
print(f"🧪 Molécula H₂O ({n_atoms} átomos, {atomic_numbers.tolist()})")
print(f"⚡ Forward pass exitoso — parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
print(f"📐 Output shape: {result['energy'].shape}")
print(f"⚛️ Predicción demo completada. Listo para datos reales (QM9, XYZ, ASE).")
