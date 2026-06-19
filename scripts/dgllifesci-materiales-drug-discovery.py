# ============================================================
# open-ai-toolkit | Descubridor de Fármacos (DGL-LifeSci)
# Rubro: Nanotecnología y Materiales
# Para qué: Molécula → toxicidad, solubilidad, binding affinity
# Demo: dataset MoleculeNet | Datos propios: 500-1000 moléculas
# ============================================================

!pip install dgl dgllife -q
from dgllife.model import GCNPredictor
import torch
import dgl

# ── Crear grafo molecular demo (anillo de 4 átomos) ─────────
g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]))
g = dgl.add_self_loop(g)

# 74 features atómicos (estándar DGL-LifeSci)
n_atoms = 4
node_feats = torch.randn(n_atoms, 74)

# ── Predecir toxicidad ──────────────────────────────────────
model = GCNPredictor(in_feats=74, n_tasks=1)
model.eval()

with torch.no_grad():
    prediction = model(g, node_feats)
    prob = torch.sigmoid(prediction).item()

print(f"✅ GCN cargado con {sum(p.numel() for p in model.parameters()):,} parámetros")
print(f"🧪 Grafo molecular: {n_atoms} átomos, {g.num_edges()} enlaces (demo)")
print(f"🔬 Features atómicos: {node_feats.shape[1]} descriptores por átomo")
print(f"⚠️  Predicción toxicidad (demo, datos sintéticos): {prob:.4f} (0=seguro, 1=tóxico)")
print(f"📊 Dataset MoleculeNet compatible (Tox21, HIV, BBBP)")
print(f"📎 Para moléculas reales: dgllife.utils.load_molecule('tu_molecula.sdf')")
