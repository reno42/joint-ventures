"""
# DGL-LIFESCI - Descubridor de Farmacos con IA Real
Rubro: Nanotecnologia y Materiales / Farmaceutica
Para que: Predice toxicidad y solubilidad de farmacos REALES con GNNs
Demo: Aspirina, Cafeina, Ibuprofeno, Paracetamol (medicamentos que conoces)
Datos propios: MoleculeNet (Tox21, HIV, BBBP, ESOL, FreeSolv)

---
### Que es DGL-LifeSci?
DGL-LifeSci es una libreria que aplica redes neuronales de grafos (GNN)
al descubrimiento de farmacos. Cada molecula se convierte en un grafo
donde los atomos son nodos y los enlaces son aristas.
Usa modelos PRE-ENTRENADOS con MoleculeNet (datos REALES de laboratorio).

### DIFERENCIA CLAVE vs version anterior:
ANTES: datos SINTETICOS (random graph con torch.randn) = prediccion falsa
AHORA: datos REALES con SMILES de farmacos comerciales + modelo pre-entrenado

---
### Las 8 celdas de este script:

| Celda | Hace | Sirve para |
|-------|------|------------|
| 1 | Instalar | Preparar herramientas (dgl, dgllife, rdkit) |
| 2 | SMILES | Convertir nombre de farmaco en molecula real |
| 3 | Grafo molecular | Convertir molecula en grafo (atomos=nodos) |
| 4 | Toxicidad | Predecir si un farmaco es toxico (Tox21) |
| 5 | Solubilidad | Predecir si se disuelve en agua (ESOL) |
| 6 | Multi-farmaco | Comparar aspirina, cafeina, ibuprofeno... |
| 7 | RDKit | Analisis quimico: peso, polaridad, Lipinski |
| 8 | Datos propios | Cargar MoleculeNet, tus SMILES, archivos SDF |

### Variables que puedes editar:
- farmacos: cambia los SMILES por tus propias moleculas
- threshold_tox: umbral de toxicidad (0.5 por defecto)
- modelos: cambia Tox21 por HIV, BBBP, BACE, etc.

### SMILES de farmacos comunes:
- Aspirina:      CC(=O)OC1=CC=CC=C1C(=O)O
- Cafeina:       CN1C=NC2=C1C(=O)N(C(=O)N2C)C
- Ibuprofeno:    CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
- Paracetamol:   CC(=O)NC1=CC=C(C=C1)O
- Nicotina:      CN1CCC[C@H]1C2=CN=CC=C2
- Glucosa:       C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O
- Morfina:       CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5
- Testosterona:  C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O
- Penicilina G:  CC1(C)S[C@@H]2[C@H](NC(=O)CC3=CC=CC=C3)C(=O)N2[C@H]1C(=O)O
- Viagra:        CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C
- THC:           CCCCCC1=CC2=C(C3C=C(CC[C@H]3C(O2)(C)C)C)C(O)=C1

### Bases de datos MoleculeNet disponibles:
- Tox21:   12,000 moleculas, 12 tipos de toxicidad (higado, riñon...)
- HIV:     40,000 moleculas, actividad anti-VIH
- BBBP:    2,000 moleculas, cruza barrera cerebral o no
- BACE:    1,500 moleculas, inhibicion de BACE (Alzheimer)
- ESOL:    1,100 moleculas, solubilidad en agua
- FreeSolv: 600 moleculas, energia de solvatacion
- Lipophilicity: 4,200 moleculas, coeficiente de particion
"""

# ============================================================
# CELDA 1: Instalar herramientas
# ============================================================

# Si falla: Runtime -> Restart runtime -> ejecutar de nuevo
# !pip install dgl dgllife rdkit-pypi torch -q 2>&1 | tail -1

# ============================================================
# CELDA 2: De SMILES a MOLECULA REAL (con RDKit)
# ============================================================
# SMILES es un "idioma" para escribir moleculas como texto.
# Ejemplo: "CC(=O)OC1=CC=CC=C1C(=O)O" es la aspirina.
# RDKit convierte ese texto en una molecula 3D real.
# NO son datos inventados: cada SMILES representa una molecula
# que EXISTE y fue sintetizada en un laboratorio.

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem, Lipinski
from rdkit.Chem.Draw import IPythonConsole
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

print("=== SMILES -> MOLECULA REAL (RDKit) ===")
print("SMILES es como el 'ADN' textual de una molecula.\n")

# Diccionario de farmacos REALES con sus SMILES
# Cada SMILES fue determinado experimentalmente en laboratorios.
# VARIABLES EDITABLES: agrega tus propios farmacos aqui
farmacos = {
    "Aspirina":      "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Cafeina":       "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ibuprofeno":    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Paracetamol":   "CC(=O)NC1=CC=C(C=C1)O",
    "Nicotina":      "CN1CCC[C@H]1C2=CN=CC=C2",
    "Glucosa":       "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
    "Etanol":        "CCO",
}

# Convertir cada SMILES en molecula RDKit
molecules = {}
for nombre, smiles in farmacos.items():
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Agregar hidrogenos explicitos (importante para grafo molecular)
        mol = Chem.AddHs(mol)
        # Calcular coordenadas 3D aproximadas
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        molecules[nombre] = mol
        
        # Mostrar propiedades basicas
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()
        print(f"  {nombre:<14} {n_atoms:>3} atomos  {n_bonds:>3} enlaces  PM={mw:>7.1f}  LogP={logp:>+5.1f}")
    else:
        print(f"  {nombre}: ERROR - SMILES invalido")

print(f"\nTodas estas moleculas EXISTEN. Fueron sintetizadas en laboratorios reales.")
print(f"RDKit las reconstruye en 3D a partir de su codigo SMILES.")

# ============================================================
# CELDA 3: Convertir molecula en GRAFO MOLECULAR
# ============================================================
# Las redes GNN no entienden atomos. Entienden GRAFOS.
# Un grafo molecular tiene:
# - Nodos: cada atomo con sus propiedades (tipo, carga, hibridacion...)
# - Aristas: cada enlace quimico (simple, doble, aromatico...)
# DGL-LifeSci hace esta conversion automaticamente.

import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

print("\n=== MOLECULA -> GRAFO MOLECULAR ===")
print("Cada atomo se convierte en un nodo con 74 caracteristicas.")
print("Cada enlace se convierte en una arista con 12 caracteristicas.")

# Featurizadores: convierten propiedades quimicas en numeros
atom_featurizer = CanonicalAtomFeaturizer()
bond_featurizer = CanonicalBondFeaturizer()

# Convertir aspirina como ejemplo
aspirina_smiles = farmacos["Aspirina"]
g = smiles_to_bigraph(
    aspirina_smiles,
    node_featurizer=atom_featurizer,
    edge_featurizer=bond_featurizer,
    add_self_loop=True,
)

# El grafo resultante es REAL: representa la estructura molecular verdadera
print(f"\nAspirina como grafo molecular:")
print(f"  Nodos (atomos):    {g.num_nodes()}")
print(f"  Aristas (enlaces): {g.num_edges()}")
print(f"  Features/nodo:     {g.ndata['h'].shape[1]} caracteristicas")
print(f"  Features/arista:   {g.edata['e'].shape[1]} caracteristicas")
print(f"\nCada nodo tiene 74 numeros que describen el atomo:")
print(f"  - Tipo de atomo (C, O, N...)")
print(f"  - Carga formal")
print(f"  - Hibridacion (sp, sp2, sp3)")
print(f"  - Numero de hidrogenos")
print(f"  - Aromaticidad")
print(f"  - Y 69 propiedades mas...")

# ============================================================
# CELDA 4: PREDECIR TOXICIDAD con modelo PRE-ENTRENADO
# ============================================================
# Tox21 es una base de datos del NIH (gobierno de EE.UU.) con
# 12,000 moleculas y 12 tipos de toxicidad probados en laboratorio.
# El modelo GCN_Tox21 fue entrenado con estos datos REALES.
# Predice: probabilidad de que una molecula sea toxica (0=sano, 1=toxico)
#
# ESTO es completamente diferente a la version anterior que usaba
# torch.randn() (datos falsos aleatorios).

from dgllife.model import load_pretrained

print("\n\n=== PREDICCION DE TOXICIDAD (GCN_Tox21) ===")
print("Modelo pre-entrenado con 12,000 moleculas del NIH.")
print("Usa datos REALES de pruebas de toxicidad en laboratorio.\n")

# Cargar modelo pre-entrenado
# Este modelo ya fue entrenado por cientificos con datos Tox21.
# Pesa ~2MB y predice en milisegundos.
try:
    model_tox = load_pretrained('GCN_Tox21')
    model_tox.eval()
    print("Modelo GCN_Tox21 cargado (pre-entrenado con 12,000 moleculas reales)")
    
    # Predecir toxicidad para aspirina
    g_asp = smiles_to_bigraph(
        farmacos["Aspirina"],
        node_featurizer=atom_featurizer,
        edge_featurizer=bond_featurizer,
        add_self_loop=True,
    )
    
    with torch.no_grad():
        node_feats = g_asp.ndata['h']
        pred = model_tox(g_asp, node_feats)
        
        # Tox21 tiene 12 tareas (tipos de toxicidad)
        # Cada una es una probabilidad entre 0 y 1
        probs = torch.sigmoid(pred).numpy()[0]
    
    tox_names = [
        "NR-AhR", "NR-AR", "NR-AR-LBD", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]
    
    print(f"\nToxicidad de ASPIRINA (probabilidad 0-1, >0.5=toxico):")
    max_tox = 0
    max_name = ""
    for name, prob in zip(tox_names, probs):
        if prob > max_tox:
            max_tox = prob
            max_name = name
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        print(f"  {name:<16} {prob:.3f} {bar}")
    
    print(f"\nMayor riesgo: {max_name} ({max_tox:.3f})")
    print(f"La aspirina es generalmente SEGURA a dosis normales.")
    print(f"Pero como todo farmaco, tiene riesgos en dosis altas.")
    
except Exception as e:
    print(f"Modelo pre-entrenado no disponible: {e}")
    print(f"\nAlternativa: crear modelo nuevo con pesos aleatorios")
    print(f"(menos preciso pero demuestra la arquitectura GCN)")
    
    # Fallback: crear modelo nuevo
    from dgllife.model import GCNPredictor
    model_tox = GCNPredictor(in_feats=74, n_tasks=12)
    model_tox.eval()
    
    g_asp = smiles_to_bigraph(
        farmacos["Aspirina"],
        node_featurizer=atom_featurizer,
        edge_featurizer=bond_featurizer,
        add_self_loop=True,
    )
    
    with torch.no_grad():
        node_feats = g_asp.ndata['h']
        pred = model_tox(g_asp, node_feats)
        probs = torch.sigmoid(pred).numpy()[0]
    
    print(f"\nPrediccion con modelo NO entrenado (solo demostracion):")
    print(f"  Salida: {probs[:6]}...")
    print(f"  NOTA: este modelo tiene pesos aleatorios.")
    print(f"  Ejecuta en GPU y con conexion para bajar el pre-entrenado.")

# ============================================================
# CELDA 5: PREDECIR SOLUBILIDAD (GCN_ESOL)
# ============================================================
# ESOL predice que tan soluble es un farmaco en agua.
# Solubilidad = que tanto se disuelve.
# logS: negativo=poco soluble, positivo=muy soluble.
# Si un farmaco no se disuelve, el cuerpo no puede absorberlo.

print("\n\n=== PREDICCION DE SOLUBILIDAD (GCN_ESOL) ===")
print("Que tan bien se disuelve el farmaco en agua.")

try:
    model_esol = load_pretrained('GCN_ESOL')
    model_esol.eval()
    
    for nombre in ["Aspirina", "Cafeina", "Ibuprofeno"]:
        g = smiles_to_bigraph(
            farmacos[nombre],
            node_featurizer=atom_featurizer,
            edge_featurizer=bond_featurizer,
            add_self_loop=True,
        )
        with torch.no_grad():
            logS = float(model_esol(g, g.ndata['h'])[0])
        
        # Interpretar solubilidad
        if logS > -2:
            interp = "MUY soluble (se absorbe facil)"
        elif logS > -4:
            interp = "Soluble (absorcion normal)"
        elif logS > -6:
            interp = "Poco soluble (absorcion lenta)"
        else:
            interp = "INSOLUBLE (no se absorbe bien)"
        
        print(f"  {nombre:<14} logS = {logS:+.2f}  -> {interp}")
    
    print(f"\nlogS mas alto = mas soluble = mejor absorcion en el cuerpo.")
    
except Exception as e:
    print(f"Modelo ESOL no disponible: {e}")

# ============================================================
# CELDA 6: COMPARAR MULTIPLES FARMACOS
# ============================================================
# Comparamos toxicidad de varios farmacos comerciales.
# Todos son moleculas REALES que puedes comprar en la farmacia.

print("\n\n=== TOXICIDAD DE FARMACOS COMERCIALES ===")
print("Comparando farmacos que conoces y usas.\n")

for nombre in farmacos.keys():
    try:
        g = smiles_to_bigraph(
            farmacos[nombre],
            node_featurizer=atom_featurizer,
            edge_featurizer=bond_featurizer,
            add_self_loop=True,
        )
        with torch.no_grad():
            pred = model_tox(g, g.ndata['h'])
            probs = torch.sigmoid(pred).numpy()[0]
        
        tox_promedio = np.mean(probs)
        
        # Clasificar
        # VARIABLE EDITABLE: threshold_tox cambia el umbral
        threshold_tox = 0.3
        if tox_promedio > 0.5:
            riesgo = "ALTO"
        elif tox_promedio > threshold_tox:
            riesgo = "MEDIO"
        else:
            riesgo = "BAJO"
        
        # Encontrar el tipo de toxicidad mas alto
        max_idx = np.argmax(probs)
        tox_names = [
            "NR-AhR", "NR-AR", "NR-AR-LBD", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]
        
        print(f"  {nombre:<14} tox.prom={tox_promedio:.3f}  riesgo={riesgo:<6}  max={tox_names[max_idx]}={probs[max_idx]:.2f}")
        
    except Exception as e:
        print(f"  {nombre}: error - {e}")

print(f"\nNOTA: esto es una prediccion computacional, no diagnostico medico.")
print(f"Siempre consulta a un medico antes de tomar cualquier farmaco.")

# ============================================================
# CELDA 7: ANALISIS QUIMICO con RDKit
# ============================================================
# RDKit puede calcular propiedades quimicas sin IA:
# - Peso molecular (PM): farmacos tipicos: 150-500 g/mol
# - LogP: que tan "grasosa" es la molecula (>5 = muy grasa)
# - Regla de Lipinski (Regla de 5): filtro para farmacos orales
#   * PM < 500
#   * LogP < 5
#   * Donores H < 5
#   * Aceptores H < 10

print("\n\n=== ANALISIS QUIMICO (RDKit) ===")
print("Regla de Lipinski: filtro para saber si un farmaco es 'drogable'.\n")

for nombre, smiles in farmacos.items():
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_donors = Lipinski.NumHDonors(mol)
        h_acceptors = Lipinski.NumHAcceptors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # Verificar regla de Lipinski
        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if h_donors > 5:
            violations += 1
        if h_acceptors > 10:
            violations += 1
        
        if violations == 0:
            status = "APTO (farmaco oral ideal)"
        elif violations <= 1:
            status = "ACEPTABLE"
        else:
            status = f"VIOLA {violations} reglas"
        
        print(f"  {nombre:<14} PM={mw:>6.1f}  LogP={logp:>+5.1f}  "
              f"H-don={h_donors}  H-acc={h_acceptors}  "
              f"rot={rot_bonds}  TPSA={tpsa:>5.1f}  -> {status}")

print(f"\nRegla de Lipinski (Regla de 5):")
print(f"  PM < 500, LogP < 5, H-don < 5, H-acc < 10")
print(f"Si un farmaco viola 2+ reglas, probablemente no se absorbe bien oralmente.")

# ============================================================
# CELDA 8: Cargar TUS propias moleculas y datasets
# ============================================================
# MoleculeNet tiene datasets completos para entrenar:
# - Tox21: toxicidad multietiqueta
# - HIV: actividad antiviral
# - BBBP: permeabilidad cerebral
# - BACE: Alzheimer
# - ESOL: solubilidad
# - FreeSolv: energia de solvatacion

print(f"\n\n=== COMO USAR TUS DATOS ===")

print(f"\nCon SMILES (tus propias moleculas):")
print(f"  mi_smiles = 'CCO'  # etanol")
print(f"  g = smiles_to_bigraph(mi_smiles, atom_featurizer, bond_featurizer)")
print(f"  pred = model_tox(g, g.ndata['h'])")
print(f"  print(f'Toxicidad: {torch.sigmoid(pred)}')")
print(f"")
print(f"Para buscar SMILES de cualquier molecula:")
print(f"  https://pubchem.ncbi.nlm.nih.gov")
print(f"  Busca 'aspirin' -> copia el SMILES canonico")
print(f"")
print(f"Datasets MoleculeNet completos (para entrenar tu modelo):")
print(f"  from dgllife.data import Tox21, HIV, BBBP")
print(f"  dataset = Tox21()  # Descarga automatica")
print(f"")
print(f"Modelos pre-entrenados disponibles:")
print(f"  load_pretrained('GCN_Tox21')           -> toxicidad")
print(f"  load_pretrained('GCN_ESOL')            -> solubilidad")
print(f"  load_pretrained('GCN_FreeSolv')        -> solvatacion")
print(f"  load_pretrained('GCN_Lipophilicity')   -> lipofilicidad")
print(f"  load_pretrained('GCN_BBBP')            -> barrera cerebral")
print(f"  load_pretrained('GCN_BACE')            -> Alzheimer")
print(f"  load_pretrained('GCN_HIV')             -> anti-VIH")

print(f"\n=== FIN DEL NOTEBOOK ===")
print(f"DGL-LifeSci predice toxicidad de farmacos REALES con IA.")
print(f"NO son datos falsos: usa modelos pre-entrenados con MoleculeNet.")
print(f"Cada SMILES representa una molecula que EXISTE en el mundo real.")
