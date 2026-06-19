# ⚛️ Nano / Materiales — 7 Herramientas en Colab

---

## 1. PyMatGen — Biblioteca de Materiales (SIN GPU) ⭐

```python
# CELDA 1
!pip install pymatgen -q
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure

# CELDA 2: Conectarse a Materials Project (API key gratis)
# Registrate en materialsproject.org → Dashboard → API Key
API_KEY = "TU_API_KEY_AQUI"  # ← PONÉ TU KEY (es gratis)
with MPRester(API_KEY) as mpr:
    # Buscar materiales con litio (baterías)
    results = mpr.summary.search(
        chemsys="Li-O",
        fields=["material_id", "formula_pretty", "band_gap", "energy_above_hull"],
        num_results=5
    )
    for r in results:
        print(f"🔬 {r.formula_pretty} | Band gap: {r.band_gap:.2f} eV | Stable: {r.energy_above_hull:.3f}")
    print(f"\n📊 Materials Project: 150,000+ materiales, API gratis")
```

**Tiempo:** 1 min | **GPU:** NO necesita | **Dataset:** Materials Project (150k+ gratis)

---

## 2. MatGL — Graph Neural Networks para Materiales

```python
# CELDA 1
!pip install matgl -q
import matgl
from matgl.ext.ase import Relaxer
import numpy as np

# CELDA 2: Predecir propiedades de un material
from pymatgen.core import Structure, Lattice

# Crear estructura Li2O simple
lattice = Lattice.cubic(4.6)
structure = Structure(lattice, ["Li", "Li", "O"], [[0,0,0], [0.25,0.25,0.25], [0.5,0.5,0.5]])

# M3GNet pre-entrenado (incluido en matgl)
from matgl.models import M3GNet
model = M3GNet.load()
print(f"✅ M3GNet cargado. Predice energía, fuerzas, stress desde estructura cristalina.")
print(f"📐 Estructura: {structure.formula}")
```

**Tiempo:** 2 min | **GPU:** Recomendada | **Dataset:** Materials Project

---

## 3. DeepMD-kit — Potenciales Interatómicos con DL

```python
# CELDA 1
!pip install deepmd-kit -q

# CELDA 2
print("✅ DeepMD-kit instalado")
print("⚛️ Reemplaza simulaciones DFT (costosas) con redes neuronales (rápidas)")
print("📊 Flujo:")
print("  1. Generar datos con DFT (VASP, Quantum ESPRESSO)")
print("  2. Entrenar DeepMD con esos datos")
print("  3. Ejecutar simulaciones 1000x más rápido con el modelo entrenado")
print("\n🔗 Casos: baterías, catálisis, materiales 2D, agua, aleaciones")
```

**Tiempo:** 2 min | **GPU:** Necesaria | **Dataset:** Cálculos DFT (VASP/QE)

---

## 4. MACE — Modelos Equivariantes (next-gen)

```python
# CELDA 1
!pip install mace-torch -q
from mace.calculators import MACECalculator
import ase
from ase import Atoms

# CELDA 2: Crear molécula de agua y predecir energía
water = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
calc = MACECalculator(model_path='medium', device='cpu')
water.set_calculator(calc)
energy = water.get_potential_energy()

print(f"💧 Energía del agua: {energy:.4f} eV")
print("✅ MACE: más rápido y preciso que DeepMD. Respeta simetrías de la física.")
```

**Tiempo:** 2 min | **GPU:** Opcional | **Dataset:** Estructuras atómicas (xyz)

---

## 5. CHGNet — GNN para Baterías

```python
# CELDA 1
!pip install chgnet -q
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from pymatgen.core import Structure, Lattice

# CELDA 2
model = CHGNet.load()
print("✅ CHGNet cargado")
print("🔋 Especializado en baterías de Li-ion")
print("   - Predice carga atómica (importante para intercalación de Li)")
print("   - Diseño de cátodos y electrolitos")
print("   - Entrenado en Materials Project + datos de baterías")
```

**Tiempo:** 2 min | **GPU:** Recomendada | **Dataset:** Estructuras de baterías

---

## 6. SchNetPack — Propiedades Moleculares

```python
# CELDA 1
!pip install schnetpack -q
import schnetpack as spk
from schnetpack.data import QM9
import torch

# CELDA 2
print("✅ SchNetPack instalado")
print("🧪 Predice propiedades moleculares desde estructura 3D:")
print("   - Energía, fuerzas, dipolos, polarizabilidad")
print("📊 Dataset QM9: 134k moléculas orgánicas pequeñas")
print("\nPara entrenar:")
print("  1. qm9 = QM9('ruta/dataset')")
print("  2. Definir modelo SchNet")
print("  3. Entrenar con spk.train")
```

**Tiempo:** 2 min | **GPU:** Recomendada | **Dataset:** QM9 (incluido)

---

## 7. DGL-LifeSci — Drug Discovery con GNNs

```python
# CELDA 1
!pip install dgl dgllife -q
import dgl
from dgllife.model import GCNPredictor
import torch

# CELDA 2
model = GCNPredictor(in_feats=74, n_tasks=1)  # predicción de solubilidad
print(f"✅ DGL-LifeSci listo. Parámetros: {sum(p.numel() for p in model.parameters()):,}")
print("💊 Aplicaciones:")
print("   - Predecir solubilidad, toxicidad, binding affinity")
print("   - Virtual screening de bibliotecas de compuestos")
print("   - Generación de moléculas (drug design)")
print("📊 Dataset: MoleculeNet (incluye Tox21, HIV, BBBP, etc.)")
```

**Tiempo:** 2 min | **GPU:** Recomendada | **Dataset:** MoleculeNet (incluido)
