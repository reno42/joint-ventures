"""
# MATGL - Predictor de Propiedades de Materiales
Rubro: Nanotecnologia y Materiales
Para que: Predice energia, fuerzas, stress y propiedades de cualquier cristal
Demo: Li2O vs NaCl vs Au (tres materiales REALES comparados)
Datos propios: Archivos CIF de estructuras cristalinas

---
### Que es MatGL (M3GNet)?
M3GNet es un "potencial universal": predice propiedades de CASI cualquier
material cristalino. Fue entrenado con 187,000 estructuras del Materials Project.
Como CHGNet pero mas general: metales, ceramicos, semiconductores...
Predice energia, fuerzas atomicas y stress (presion interna).

---
### Las 7 celdas de este script:

| Celda | Hace | Sirve para |
|-------|------|------------|
| 1 | Instalar | Preparar herramientas (matgl, pymatgen) |
| 2 | Cargar M3GNet | Bajar cerebro entrenado (187k estructuras) |
| 3 | Li2O | Analizar oxido de litio (baterias, ceramica) |
| 4 | NaCl | Sal de mesa: un cristal ionico simple |
| 5 | Oro (Au) | Metal noble: compara con cristales ionicos |
| 6 | Fuerzas | Ver fuerzas atomo por atomo (quien empuja a quien) |
| 7 | Relajar | Encontrar la estructura mas estable |
| 8 | Datos propios | Cargar tus materiales desde CIF |

### Variables que puedes editar:
- Material: cambia elementos y posiciones en Structure(...)
- Parametro de red (a): tamaño de la celda cristalina
- fmax: precision de relajacion (0.1=rapido, 0.01=normal, 0.001=fino)
- fmax_relax: igual que arriba
- steps: pasos de optimizacion

### Propiedades que predice M3GNet:
- Energia (eV): estabilidad del material
- Fuerzas (eV/A): hacia donde se mueve cada atomo
- Stress (GPa): presion interna del cristal
- Con el modelo se puede: relajar, dinamica molecular, propiedades elasticas

### Aplicaciones reales:
- Baterias: disenar electrolitos solidos y nuevos catodos
- Aleaciones: crear metales mas fuertes y ligeros
- Semiconductores: nuevos materiales para chips
- Catalisis: encontrar catalizadores para reacciones quimicas
- Captura de CO2: materiales que absorben dioxido de carbono
"""

# ============================================================
# CELDA 1: Instalar herramientas
# ============================================================

# Si falla: Runtime -> Restart runtime -> ejecutar de nuevo
# !pip install matgl pymatgen ase -q 2>&1 | tail -1

# ============================================================
# CELDA 2: Cargar M3GNet pre-entrenado
# ============================================================
# M3GNet fue entrenado con 187,000 estructuras del Materials Project.
# Es un "potencial universal": funciona para metales, oxidos,
# semiconductores, ceramicos... casi todo.

import warnings
warnings.filterwarnings("ignore")

import matgl
from matgl.ext.ase import M3GNetCalculator, Relaxer
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from ase import Atoms

print("Cargando M3GNet pre-entrenado...")
print("(entrenado con 187,000 estructuras del Materials Project)")

# Cargar modelo
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
model = pot.model

print(f"M3GNet cargado! Predice: energia, fuerzas, stress")
print(f"Potencial universal: funciona para metales, oxidos, semiconductores...")

# ============================================================
# CELDA 3: Li2O - Oxido de Litio (material de bateria / ceramica)
# ============================================================
# Li2O es un oxido ionico usado en:
# - Baterias de litio-aire (el "santo grial": 10x mas densidad)
# - Ceramicas avanzadas
# - Recubrimientos protectores
# Estructura: cubica tipo antifluorita (como NaCl pero al reves)

print("\n\n=== Li2O: OXIDO DE LITIO ===")
print("Usado en baterias de litio-aire y ceramicas avanzadas.")

# VARIABLES EDITABLES:
a_li2o = 4.61   # <-- EDITAR: parametro de red cubico (Angstroms)

# Estructura cubica (antifluorita)
# Li en posiciones tetraedricas, O en posiciones fcc
lattice_li2o = Lattice.cubic(a_li2o)

structure_li2o = Structure(
    lattice_li2o,
    ["Li", "Li", "O"],   # Proporcion 2:1 = Li2O
    [
        [0.25, 0.25, 0.25],    # Li (sitio tetraedrico)
        [0.75, 0.75, 0.75],    # Li
        [0.00, 0.00, 0.00],    # O (esquina del cubo)
    ]
)

# Predecir propiedades
pred_li2o = model.predict_structure(structure_li2o)

energy_li2o = float(pred_li2o['e'])
energy_atom_li2o = energy_li2o / len(structure_li2o)
forces_li2o = pred_li2o['f']
stress_li2o = pred_li2o['s']

print(f"\nFormula:     Li2O")
print(f"Atomos:       {len(structure_li2o)}")
print(f"Estructura:  Cubica antifluorita")
print(f"Energia:      {energy_li2o:.4f} eV  ({energy_atom_li2o:.4f} eV/atomo)")
print(f"Fuerza max:   {np.max(np.abs(forces_li2o)):.4f} eV/A")
print(f"Stress (GPa): {np.mean(np.diag(stress_li2o)):.2f} (presion hidrostatica)")

# ============================================================
# CELDA 4: NaCl - Sal de Mesa (cristal ionico clasico)
# ============================================================
# NaCl es el cristal ionico mas simple y conocido.
# Sirve para calibrar: todos los metodos deberian dar resultados
# muy precisos para NaCl porque es ultra-estudiado.

print("\n\n=== NaCl: SAL DE MESA ===")
print("El cristal mas simple. Sirve para verificar que el modelo funciona.")

# VARIABLES EDITABLES:
a_nacl = 5.64   # <-- EDITAR: parametro de red (Angstroms)

lattice_nacl = Lattice.cubic(a_nacl)
structure_nacl = Structure(
    lattice_nacl,
    ["Na", "Cl"],
    [
        [0.00, 0.00, 0.00],    # Na (esquina)
        [0.50, 0.50, 0.50],    # Cl (centro)
    ]
)

pred_nacl = model.predict_structure(structure_nacl)

energy_nacl = float(pred_nacl['e'])
energy_atom_nacl = energy_nacl / len(structure_nacl)
forces_nacl = pred_nacl['f']
stress_nacl = pred_nacl['s']

print(f"\nFormula:     NaCl")
print(f"Atomos:       {len(structure_nacl)}")
print(f"Estructura:  Cubica (tipo halita)")
print(f"Energia:      {energy_nacl:.4f} eV  ({energy_atom_nacl:.4f} eV/atomo)")
print(f"Fuerza max:   {np.max(np.abs(forces_nacl)):.4f} eV/A")
print(f"Stress (GPa): {np.mean(np.diag(stress_nacl)):.2f}")

# ============================================================
# CELDA 5: Au - Oro (metal noble FCC)
# ============================================================
# El oro es un metal puro con estructura FCC (cubica centrada en caras).
# Compararlo con Li2O y NaCl muestra como M3GNet funciona igual de bien
# para todo tipo de materiales: ionicos, metalicos, covalentes.

print("\n\n=== Au: ORO METALICO ===")
print("Metal noble. M3GNet funciona para TODO tipo de enlace quimico.")

# VARIABLES EDITABLES:
a_au = 4.08    # <-- EDITAR: parametro de red del oro (Angstroms)

lattice_au = Lattice.cubic(a_au)
structure_au = Structure(
    lattice_au,
    ["Au"],
    [
        [0.00, 0.00, 0.00],    # Au (esquina FCC)
    ]
)

pred_au = model.predict_structure(structure_au)

energy_au = float(pred_au['e'])
forces_au = pred_au['f']
stress_au = pred_au['s']

print(f"\nFormula:     Au")
print(f"Atomos:       {len(structure_au)}")
print(f"Estructura:  Cubica FCC")
print(f"Energia:      {energy_au:.4f} eV  ({energy_au:.4f} eV/atomo)")
print(f"Fuerza max:   {np.max(np.abs(forces_au)):.4f} eV/A")
print(f"Stress (GPa): {np.mean(np.diag(stress_au)):.2f}")

# -- COMPARACION de los 3 materiales --
print(f"\n=== COMPARACION DE MATERIALES ===")
print(f"{'Material':<12} {'eV/atomo':<12} {'Tipo':<20} {'F.max (eV/A)':<15}")
print(f"-" * 60)
print(f"{'Li2O':<12} {energy_atom_li2o:<12.4f} {'Oxido ionico':<20} {np.max(np.abs(forces_li2o)):<15.4f}")
print(f"{'NaCl':<12} {energy_atom_nacl:<12.4f} {'Cristal ionico':<20} {np.max(np.abs(forces_nacl)):<15.4f}")
print(f"{'Au':<12} {energy_au:<12.4f} {'Metal FCC':<20} {np.max(np.abs(forces_au)):<15.4f}")
print(f"\nEnergia negativa = estable. M3GNet captura los tres tipos de enlace.")

# ============================================================
# CELDA 6: FUERZAS atomo por atomo
# ============================================================
# Las fuerzas te dicen hacia donde "quiere" moverse cada atomo.
# Fuerza ~0: atomo comodo, posicion optima.
# Fuerza grande: atomo "incomodo", la estructura no es estable.
# Esto es CLAVE para disenar materiales: quieres que todos los
# atomos tengan fuerzas cercanas a 0.

print("\n\n=== FUERZAS POR ATOMO ===")
print("Fuerza ~0 = atomo comodo. Fuerza grande = atomo quiere moverse.")

# Li2O - fuerzas por atomo
print(f"\nLi2O - fuerzas por atomo (eV/A):")
for i in range(len(structure_li2o)):
    f_mag = np.linalg.norm(forces_li2o[i])
    sym = structure_li2o[i].species_string
    fx, fy, fz = float(forces_li2o[i][0]), float(forces_li2o[i][1]), float(forces_li2o[i][2])
    print(f"  Atomo {i+1} ({sym}): |F|={f_mag:.4f}  Fx={fx:+.4f}  Fy={fy:+.4f}  Fz={fz:+.4f}")

# NaCl - fuerzas por atomo
print(f"\nNaCl - fuerzas por atomo (eV/A):")
for i in range(len(structure_nacl)):
    f_mag = np.linalg.norm(forces_nacl[i])
    sym = structure_nacl[i].species_string
    fx, fy, fz = float(forces_nacl[i][0]), float(forces_nacl[i][1]), float(forces_nacl[i][2])
    print(f"  Atomo {i+1} ({sym}): |F|={f_mag:.4f}  Fx={fx:+.4f}  Fy={fy:+.4f}  Fz={fz:+.4f}")

# Au - fuerzas por atomo
print(f"\nAu - fuerzas por atomo (eV/A):")
for i in range(len(structure_au)):
    f_mag = np.linalg.norm(forces_au[i])
    print(f"  Atomo 1 (Au): |F|={f_mag:.4f}  Fx={float(forces_au[0][0]):+.4f}  Fy={float(forces_au[0][1]):+.4f}  Fz={float(forces_au[0][2]):+.4f}")

# ============================================================
# CELDA 7: RELAJAR la estructura
# ============================================================
# Relajar = mover atomos hasta que las fuerzas sean ~0.
# La estructura inicial tiene fuerzas (atomos no estan en el optimo).
# Despues de relajar, la estructura esta en su verdadero estado base.

print("\n\n=== RELAJACION DE ESTRUCTURA ===")
print("Buscando la geometria optima...")

# VARIABLES EDITABLES:
#   fmax=0.01: precision normal
#   fmax=0.001: precision alta (mas lento)
#   steps=100: rapido, steps=500: profesional
fmax_relax = 0.05      # <-- EDITAR: max fuerza aceptable (eV/A)
steps_relax = 200      # <-- EDITAR: pasos maximos

# Relajar Li2O como ejemplo
from matgl.ext.ase import Relaxer

relaxer = Relaxer(potential=pot, relax_cell=False)  # False = no cambiar tamaño de celda

print(f"\nRelajando Li2O (fmax={fmax_relax}, steps={steps_relax})...")

# Convertir pymatgen Structure a ASE Atoms
ase_li2o = structure_li2o.to_ase_atoms()

# Relajar
relaxed = relaxer.relax(ase_li2o, fmax=fmax_relax, steps=steps_relax)
relaxed_structure = relaxed['final_structure']
trajectory = relaxed['trajectory']

print(f"\nEnergia inicial: {trajectory.energies[0]:.4f} eV")
print(f"Energia final:   {trajectory.energies[-1]:.4f} eV")
print(f"Cambio energia:  {trajectory.energies[-1] - trajectory.energies[0]:.4f} eV")
print(f"Pasos usados:    {len(trajectory.energies) - 1}")

# Mostrar cambio en posiciones
print(f"\nCambios en posiciones atomicas (Li2O):")
for i in range(len(ase_li2o)):
    old_pos = ase_li2o[i].position
    new_pos = relaxed_structure[i].position
    diff = np.linalg.norm(new_pos - old_pos)
    sym = ase_li2o[i].symbol
    print(f"  {sym}: desplazamiento = {diff:.4f} A")

print(f"\nDespues de relajar, la estructura esta en equilibrio.")
print(f"Esa es la geometria que tendria en la naturaleza real.")

# ============================================================
# CELDA 8: Cargar TUS materiales desde CIF
# ============================================================
# Descarga archivos CIF de:
# - https://materialsproject.org (gratis, 150k+ materiales)
# - https://www.crystallography.net/cod/ (gratis, 500k+ estructuras)
# - https://icsd.products.fiz-karlsruhe.de (base de datos profesional)

print(f"\n\n=== COMO USAR TUS PROPIOS MATERIALES ===")

print(f"Ejemplo:")
print(f"  from pymatgen.core import Structure")
print(f"  mi_mat = Structure.from_file('tu_material.cif')")
print(f"  pred = model.predict_structure(mi_mat)")
print(f"  print(f'Energia: {pred[\"e\"]:.4f} eV')")
print(f"  print(f'Fuerza max: {np.max(np.abs(pred[\"f\"])):.4f} eV/A')")
print(f"  print(f'Stress: {pred[\"s\"]} GPa')")
print(f"")
print(f"IDs interesantes en Materials Project:")
print(f"  LiCoO2:    mp-22526  (catodo de bateria)")
print(f"  LiFePO4:   mp-19017  (catodo Tesla)")
print(f"  SiO2:      mp-6930   (cuarzo)")
print(f"  TiO2:      mp-2657   (dioxido de titanio)")
print(f"  ZnO:       mp-2133   (oxido de zinc)")
print(f"  Si:        mp-149    (silicio, chip)")
print(f"  Cu:        mp-30     (cobre)")
print(f"  Al:        mp-134    (aluminio)")

print(f"\n=== FIN DEL NOTEBOOK ===")
print(f"M3GNet es un potencial universal para TODO tipo de cristales.")
print(f"Predice energia, fuerzas y stress de materiales reales.")
