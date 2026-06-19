# ============================================================
# open-ai-toolkit | Simulador Atómico Avanzado (MACE)
# Rubro: Nanotecnología y Materiales
# Para qué: Predice energía atómica con precisión cuántica
# Demo: molécula H2O | Datos propios: archivos XYZ
# ============================================================

!pip install mace-torch ase -q
from mace.calculators import MACECalculator
from ase import Atoms

water = Atoms('H2O', positions=[[0,0,0],[0.96,0,0],[-0.24,0.93,0]])
calc = MACECalculator(model_path='medium', device='cpu')
water.set_calculator(calc)

print(f"💧 Energía del agua: {water.get_potential_energy():.4f} eV")
print("✅ MACE: más rápido que DeepMD. Respeta simetrías físicas.")
