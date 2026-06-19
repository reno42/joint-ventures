# ============================================================
# open-ai-toolkit | Analizador de Redes (NetworkX + Node2Vec)
# Rubro: Ciencias Sociales
# Para qué: Lista de conexiones → quién influye, grupos de poder
# Demo: red sintética | Datos propios: CSV (origen, destino)
# ============================================================

!pip install networkx node2vec -q
import networkx as nx, numpy as np
from node2vec import Node2Vec
import matplotlib.pyplot as plt

G = nx.Graph()
personas = ['Ana','Bruno','Carla','Diego','Elena']
comisiones = ['Economía','Salud','Educación','Defensa']
for p in personas:
    for c in np.random.choice(comisiones, 2, replace=False):
        G.add_edge(p, c)

n2v = Node2Vec(G, dimensions=16, walk_length=5, num_walks=50)
model = n2v.fit(window=3, min_count=1)

target = 'Ana'
print(f"👥 Cercanos a {target}:")
for name, score in model.wv.most_similar(target):
    print(f"  {name}: {score:.2f}")

nx.draw(G, with_labels=True, node_color='lightblue')
plt.title("Red de Influencia")
plt.show()
