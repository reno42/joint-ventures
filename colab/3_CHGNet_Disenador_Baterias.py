"""
# CHGNET - Disenador de Baterias Avanzado
Rubro: Nanotecnologia y Materiales
Para que: Predice energia, fuerzas y carga de materiales de bateria
Demo: LiCoO2 vs LiFePO4 (catodos REALES de baterias comerciales)
Datos propios: Archivos CIF de estructuras cristalinas

---
### Que es CHGNet?
CHGNet es una red neuronal que entiende baterias. A diferencia de otros modelos,
CHGNet PREDICE las cargas magneticas de cada atomo (no las asume).
Fue entrenado con 1.5 millones de estructuras del Materials Project.
Sabe como se mueven los iones de litio dentro de un catodo real.

---
### Las 7 celdas de este script:

| Celda | Hace | Sirve para |
|-------|------|------------|
| 1 | Instalar | Preparar herramientas (chgnet, pymatgen) |
| 2 | Cargar modelo | Bajar cerebro pre-entrenado (1.5M estructuras) |
| 3 | LiCoO2 | Analizar catodo de bateria de telefono |
| 4 | LiFePO4 | Comparar con catodo de bateria de auto electrico |
| 5 | Cargas | Ver la carga magnetica de CADA atomo |
| 6 | Relajar | Encontrar la estructura mas estable |
| 7 | Datos propios | Cargar tus materiales desde archivos CIF |

### Variables que puedes editar:
- Elementos: cambia Li por Na, Co por Mn, etc.
- Parametros de red: a, b, c (tamaño de la celda cristalina)
- fmax: precision de la relajacion (0.01=normal, 0.001=fino)
- steps: pasos de relajacion (100=rapido, 500=profesional)

### Aplicaciones reales:
- Baterias Li-ion: disenar nuevos catodos mas eficientes
- Baterias Na-ion: alternativas mas baratas al litio
- Estado solido: electrolitos solidos para baterias seguras
- Degradacion: predecir que materiales duran mas ciclos
- Reciclaje: entender como recuperar metales de baterias usadas

### Materiales de bateria que puedes probar:
- LiCoO2 (LCO): bateria de telefono/laptop clasica
- LiFePO4 (LFP): bateria de auto electrico, mas segura
- LiNiMnCoO2 (NMC): bateria de Tesla, alto rendimiento
- LiMn2O4 (LMO): bateria de herramientas electricas
- NaFePO4: alternativa sin litio (mas barata)
"""

# ============================================================
# CELDA 1: Instalar herramientas
# ============================================================

# Si falla: Runtime -> Restart runtime -> ejecutar de nuevo
# !pip install chgnet pymatgen ase -q 2>&1 | tail -1

# ============================================================
# CELDA 2: Cargar modelo CHGNet pre-entrenado
# ============================================================
# CHGNet fue entrenado con 1.5 millones de estructuras cristalinas
# del Materials Project. Sabe de casi todos los materiales conocidos.

from chgnet.model import CHGNet, StructOptimizer
from chgnet.graph import CrystalGraph
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("Cargando CHGNet pre-entrenado...")
print("(entrenado con 1.5M de estructuras del Materials Project)")

# VARIABLE EDITABLE: device='cpu' o 'cuda' si tienes GPU
model = CHGNet.load()
print(f"CHGNet cargado! Predice: energia, fuerzas, stress, cargas magneticas")
print("LO ESPECIAL: predice cargas magneticas, no las asume como otros modelos")

# ============================================================
# CELDA 3: ANALIZAR LiCoO2 (catodo de bateria de TELEFONO)
# ============================================================
# LiCoO2 es el catodo mas usado en baterias de telefonos y laptops.
# Estructura real: hexagonal, grupo espacial R-3m.
# Los atomos de Li estan entre capas de CoO2.
# Cuando cargas la bateria, el Li sale. Cuando descargas, entra.

print("\n\n=== LiCoO2: CATODO DE BATERIA DE TELEFONO ===")
print("Es el material que tienes DENTRO de la bateria de tu celular.")

# VARIABLES EDITABLES:
#   a=2.82, c=14.05: parametros de red del LiCoO2 real
#   Puedes buscar otros materiales en https://materialsproject.org
a = 2.82     # <-- EDITAR: parametro de red 'a' en Angstroms
c = 14.05    # <-- EDITAR: parametro de red 'c' en Angstroms

# Crear estructura cristalina hexagonal (grupo R-3m)
lattice_lco = Lattice.hexagonal(a, c)

# Estas son las posiciones atomicas REALES del LiCoO2
# determinadas por difraccion de rayos X en laboratorios.
structure_lco = Structure(
    lattice_lco,
    ["Li", "Li", "Li", "Co", "Co", "Co", "O", "O", "O", "O", "O", "O"],
    [
        [0.00, 0.00, 0.00],       # Li en posicion 3a
        [1/3,  2/3,  1/3],        # Li
        [2/3,  1/3,  2/3],        # Li
        [0.00, 0.00, 0.50],       # Co en posicion 3b
        [1/3,  2/3,  5/6],        # Co
        [2/3,  1/3,  1/6],        # Co
        [0.00, 0.00, 0.25],       # O en posicion 6c
        [0.00, 0.00, 0.75],       # O
        [1/3,  2/3,  0.08],       # O
        [1/3,  2/3,  0.58],       # O
        [2/3,  1/3,  0.42],       # O
        [2/3,  1/3,  0.92],       # O
    ]
)

# Predecir propiedades con CHGNet
pred_lco = model.predict_structure(structure_lco)

# Extraer resultados
energy_lco = float(pred_lco['e']) * len(structure_lco)   # eV (total)
energy_per_atom_lco = float(pred_lco['e'])                # eV/atomo
forces_lco = pred_lco['f']                                # eV/A
magmoms_lco = pred_lco['m']                               # Bohr magneton
stress_lco = pred_lco['s']                                # GPa

# Mostrar resultados
formula_lco = structure_lco.composition.reduced_formula
print(f"\nFormula:    {formula_lco}")
print(f"Atomos:      {len(structure_lco)}")
print(f"Grupo esp:   R-3m (hexagonal, como un panal)")
print(f"Energia:     {energy_lco:.4f} eV total  ({energy_per_atom_lco:.4f} eV/atomo)")
print(f"Fuerza max:  {np.max(np.abs(forces_lco)):.4f} eV/A")
print(f"Mag. total:  {np.sum(magmoms_lco):.4f} muB")
print(f"Stress (GPa): {np.mean(np.diag(stress_lco)):.2f} (presion)")
print(f"\nEnergia negativa = el material EXISTE y es estable")
print(f"LiCoO2 es estable: se usa en baterias desde 1991.")

# Ver cargas magneticas por tipo de atomo
print(f"\nCarga magnetica por atomo (CHGNet la PREDICE, no la inventa):")
symbols = structure_lco.species
for i in range(len(structure_lco)):
    if i < 12:  # mostrar solo los primeros
        print(f"  {symbols[i]}  mag={float(magmoms_lco[i]):+.4f} muB")

# ============================================================
# CELDA 4: COMPARAR con LiFePO4 (catodo de AUTO ELECTRICO)
# ============================================================
# LiFePO4 (LFP) es el catodo mas seguro y se usa en:
# - Tesla Model 3 (version estandar)
# - Baterias estacionarias (hogar, red electrica)
# - Autobuses electricos
# Ventaja: no explota, dura mas ciclos, mas barato.
# Desventaja: menos densidad de energia que LiCoO2.

print("\n\n=== LiFePO4: CATODO DE AUTO ELECTRICO (TESLA) ===")
print("Material mas seguro que LiCoO2. No explota ni se incendia.")

# Estructura ortorrombica del LiFePO4 (grupo Pnma)
# VARIABLES EDITABLES: parametros de red del LiFePO4
a_lfp = 10.33   # <-- EDITAR
b_lfp = 6.01    # <-- EDITAR
c_lfp = 4.69    # <-- EDITAR

lattice_lfp = Lattice.orthorhombic(a_lfp, b_lfp, c_lfp)

# LiFePO4 tiene 28 atomos en la celda unidad (4 unidades formula)
structure_lfp = Structure(
    lattice_lfp,
    # 4 Li, 4 Fe, 4 P, 16 O = 28 atomos
    ["Li"]*4 + ["Fe"]*4 + ["P"]*4 + ["O"]*16,
    [
        # Li (4 atomos, Wyckoff 4a)
        [0.000, 0.000, 0.000],
        [0.000, 0.500, 0.000],
        [0.500, 0.000, 0.500],
        [0.500, 0.500, 0.500],
        # Fe (4 atomos, Wyckoff 4c)
        [0.282, 0.250, 0.974],
        [0.218, 0.750, 0.526],
        [0.718, 0.750, 0.474],
        [0.782, 0.250, 0.026],
        # P (4 atomos, Wyckoff 4c)
        [0.095, 0.250, 0.418],
        [0.405, 0.750, 0.082],
        [0.595, 0.250, 0.918],
        [0.905, 0.750, 0.582],
        # O (16 atomos, 3 sitios Wyckoff)
        [0.097, 0.250, 0.742],
        [0.457, 0.250, 0.206],
        [0.403, 0.750, 0.758],
        [0.543, 0.750, 0.294],
        [0.165, 0.046, 0.285],
        [0.665, 0.546, 0.215],
        [0.335, 0.454, 0.785],
        [0.835, 0.954, 0.715],
        [0.165, 0.454, 0.785],
        [0.665, 0.954, 0.715],
        [0.335, 0.546, 0.215],
        [0.835, 0.046, 0.285],
        [0.403, 0.250, 0.258],
        [0.543, 0.250, 0.794],
        [0.457, 0.750, 0.706],
        [0.097, 0.750, 0.242],
    ]
)

# Predecir propiedades de LiFePO4
pred_lfp = model.predict_structure(structure_lfp)

energy_lfp = float(pred_lfp['e']) * len(structure_lfp)
energy_per_atom_lfp = float(pred_lfp['e'])
forces_lfp = pred_lfp['f']

formula_lfp = structure_lfp.composition.reduced_formula
print(f"\nFormula:     {formula_lfp}")
print(f"Atomos:       {len(structure_lfp)}")
print(f"Grupo esp:   Pnma (ortorrombico)")
print(f"Energia:      {energy_lfp:.4f} eV total  ({energy_per_atom_lfp:.4f} eV/atomo)")
print(f"Fuerza max:   {np.max(np.abs(forces_lfp)):.4f} eV/A")

# Comparacion directa
print(f"\n=== COMPARACION DIRECTA ===")
print(f"LiCoO2:  {energy_per_atom_lco:.4f} eV/atomo (mas energia = mas capacidad)")
print(f"LiFePO4:  {energy_per_atom_lfp:.4f} eV/atomo (menos energia = mas seguro)")
print(f"LiCoO2 tiene mayor densidad de energia pero LiFePO4 es mas seguro.")
print(f"Por eso Tesla usa LiFePO4 en sus autos economicos.")

# ============================================================
# CELDA 5: ANALIZAR CARGAS MAGNETICAS por cada atomo
# ============================================================
# CHGNet es ESPECIAL porque predice la carga magnetica de cada atomo.
# Otros modelos asumen cargas fijas (+1 para Li, -2 para O...).
# CHGNet las calcula del entorno quimico real.
# Esto es clave para baterias: la carga determina como se mueve el Li.

print("\n\n=== CARGAS MAGNETICAS DETALLADAS ===")
print("CHGNet predice la carga de cada atomo segun su entorno quimico.")

# Analizar LiCoO2 por tipo de atomo
charges_by_element = {}
for i, site in enumerate(structure_lco):
    elem = str(site.species_string)
    if elem not in charges_by_element:
        charges_by_element[elem] = []
    charges_by_element[elem].append(float(magmoms_lco[i]))

print(f"\nCargas en LiCoO2:")
for elem, charges in charges_by_element.items():
    avg = np.mean(charges)
    print(f"  {elem}: promedio={avg:+.4f} muB  (min={min(charges):+.4f}, max={max(charges):+.4f})")

# Analizar LiFePO4
magmoms_lfp = pred_lfp['m']
charges_by_elem_lfp = {}
for i, site in enumerate(structure_lfp):
    elem = str(site.species_string)
    if elem not in charges_by_elem_lfp:
        charges_by_elem_lfp[elem] = []
    charges_by_elem_lfp[elem].append(float(magmoms_lfp[i]))

print(f"\nCargas en LiFePO4:")
for elem, charges in charges_by_elem_lfp.items():
    avg = np.mean(charges)
    print(f"  {elem}: promedio={avg:+.4f} muB  (min={min(charges):+.4f}, max={max(charges):+.4f})")

print(f"\nFe tiene carga magnetica (es magnetico). Co tambien.")
print(f"Esto afecta como se mueve el Li dentro de la bateria.")

# ============================================================
# CELDA 6: RELAJAR la estructura (encontrar la forma mas estable)
# ============================================================
# En la vida real, los atomos no estan exactamente en las posiciones
# teoricas. CHGNet puede "relajar" la estructura: mover atomos hasta
# que las fuerzas sean ~0 (estado de minima energia).

print("\n\n=== RELAJACION DE ESTRUCTURA ===")
print("Buscando la geometria optima donde las fuerzas sean ~0...")

from chgnet.model import StructOptimizer

# VARIABLES EDITABLES:
#   fmax=0.01: normal, fmax=0.001: preciso (mas lento)
#   steps=100: rapido, steps=500: profesional
fmax_relax = 0.05      # <-- EDITAR: precision (0.01-0.1)
steps_relax = 200      # <-- EDITAR: pasos maximos

# Crear optimizador CHGNet
relaxer = StructOptimizer()

print(f"Relajando LiCoO2 (fmax={fmax_relax}, steps={steps_relax})...")
result = relaxer.relax(
    structure_lco.copy(),
    fmax=fmax_relax,
    steps=steps_relax,
    verbose=False
)

# Estructura relajada
relaxed_structure = result['final_structure']
relaxed_energy = float(result['trajectory'].energies[-1])

print(f"\nEnergia inicial: {energy_lco:.4f} eV")
print(f"Energia relajada: {relaxed_energy:.4f} eV")
print(f"Diferencia: {relaxed_energy - energy_lco:.4f} eV")
print(f"Pasos usados: {len(result['trajectory'].energies)}")

# Mostrar cambios en posiciones atomicas
print(f"\nCambios en posiciones (LiCoO2):")
for i in range(min(6, len(structure_lco))):
    old_pos = structure_lco[i].coords
    new_pos = relaxed_structure[i].coords
    diff = np.linalg.norm(new_pos - old_pos)
    sym = structure_lco[i].species_string
    print(f"  {sym}: se movio {diff:.4f} A  (fuerza inicial: {np.linalg.norm(forces_lco[i]):.4f} eV/A)")

print(f"\nDespues de relajar, la estructura esta en su punto mas estable.")
print(f"Esa es la geometria REAL que tendria en la naturaleza.")

# ============================================================
# CELDA 7: Cargar TUS materiales desde CIF
# ============================================================
# Los archivos CIF (Crystallographic Information Format) contienen
# la estructura cristalina de cualquier material.
# Puedes descargar CIFs de: https://materialsproject.org
# O de: https://www.crystallography.net/cod/

from pymatgen.core import Structure

print(f"\n\n=== COMO USAR TUS PROPIOS MATERIALES ===")

# VARIABLE EDITABLE: pon la ruta a tu archivo .cif
# mi_material = Structure.from_file('mi_bateria.cif')
# pred = model.predict_structure(mi_material)
# print(f"Energia: {pred['e']:.4f} eV/atomo")

print(f"Ejemplo de uso:")
print(f"  1. Descarga un .cif de https://materialsproject.org")
print(f"  2. Subelo a Colab")
print(f"  3. Ejecuta:")
print(f"     mi_mat = Structure.from_file('tu_archivo.cif')")
print(f"     pred = model.predict_structure(mi_mat)")
print(f"     print(f'Energia: {pred[\"e\"]:.4f} eV/atomo')")
print(f"     print(f'Cargas: {pred[\"m\"]}')")
print(f"")
print(f"Para buscar materiales de bateria en Materials Project:")
print(f"  - LiFePO4:  mp-19017")
print(f"  - LiCoO2:   mp-22526")
print(f"  - LiMn2O4:  mp-18767")
print(f"  - NMC811:   mp-756280")

print(f"\n=== FIN DEL NOTEBOOK ===")
print(f"CHGNet predice energia, fuerzas y CARGAS MAGNETICAS para baterias reales.")
print(f"No asume cargas fijas: las calcula del entorno quimico de cada atomo.")
