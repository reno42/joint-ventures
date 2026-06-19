# 📊 Ciencias Sociales — 8 Herramientas en Colab

---

## 1. BETO — BERT en Español ⭐

```python
# CELDA 1
!pip install transformers torch -q
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# CELDA 2: Cargar BETO (BERT entrenado en español por U. de Chile)
model_name = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

# CELDA 3: Clasificar textos peruanos
textos = [
    "El proyecto de ley propone modificar el artículo 14 de la constitución para incluir derechos digitales.",
    "La inflación en Perú bajó al 2.5% este mes según el BCRP.",
    "Se reportaron protestas en el sur del país por el alza de combustibles."
]

for t in textos:
    print(f"📝 {t[:80]}...")
    print(f"   → Clasificación: {classifier(t)[0]}\n")

print("✅ BETO cargado. Fine-tuneá con tus datos de leyes/decretos peruanos.")
```

**Tiempo:** 2 min (descarga ~400MB) | **GPU:** Recomendada | **Dataset:** Textos español

---

## 2. MarIA — RoBERTa Español (BSC Barcelona)

```python
# CELDA 1
!pip install transformers -q
from transformers import pipeline

# CELDA 2: MarIA (modelos del Barcelona Supercomputing Center)
# Más grandes y mejores que BETO para tareas complejas
model_name = "PlanTL-GOB-ES/roberta-base-bne"

try:
    fill_mask = pipeline('fill-mask', model=model_name)
    result = fill_mask("La nueva ley de protección de datos en Perú <mask> los derechos de los ciudadanos.")
    for r in result:
        print(f"🔤 {r['token_str']}: {r['score']:.3f}")
except Exception as e:
    print(f"⚠️ Modelo grande para Colab free: {e}")
    print("Probá con el modelo más pequeño:")
    print("model_name = 'PlanTL-GOB-ES/roberta-base-bne'")
```

**Tiempo:** 2 min | **GPU:** Recomendada | **Dataset:** Textos español (train BNE)

---

## 3. GPT4All — LLM Offline SIN API ⭐

```python
# CELDA 1
!pip install gpt4all -q
from gpt4all import GPT4All

# CELDA 2: Descargar y cargar modelo (3-4GB, solo la primera vez)
# Modelo recomendado para español: Llama 3.1 8B
model = GPT4All("Meta-Llama-3.1-8B-Instruct-128k-GGUF")  # descarga ~4GB

# CELDA 3: Analizar documento legal
respuesta = model.generate(
    "Analizá el siguiente texto legal peruano y resumilo en 3 puntos clave:\n\n"
    "'El Decreto Supremo N° 004-2024 establece medidas para la transformación digital "
    "del Estado peruano, incluyendo la interoperabilidad de sistemas, la identidad digital "
    "única y la capacitación de servidores públicos en competencias digitales.'",
    max_tokens=300
)
print(respuesta)
print("\n✅ GPT4All: LLM corriendo OFFLINE en esta notebook. Sin API costs.")
```

**Tiempo:** 5 min (primera descarga 4GB) | **GPU:** NO (CPU) | **Dataset:** Ninguno (pre-entrenado)

---

## 4. BERTopic — Descubrimiento de Temas

```python
# CELDA 1
!pip install bertopic -q
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# CELDA 2: Dataset de ejemplo (reemplazar con tus documentos)
docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data'][:500]

# CELDA 3: Extraer temas
topic_model = BERTopic(language="multilingual", verbose=True)
topics, probs = topic_model.fit_transform(docs)

# CELDA 4: Ver resultados
freq = topic_model.get_topic_info()
print(f"📊 {len(freq)} temas descubiertos:")
for i, row in freq.head(10).iterrows():
    if row['Topic'] != -1:
        words = topic_model.get_topic(row['Topic'])
        top_words = ", ".join([w[0] for w in words[:5]])
        print(f"  Tema {row['Topic']}: {top_words} ({row['Count']} docs)")

# CELDA 5: Visualizar
topic_model.visualize_topics()
```

**Tiempo:** 3 min | **GPU:** NO | **Dataset:** Documentos de texto

---

## 5. NetworkX + Node2Vec — Análisis de Redes

```python
# CELDA 1
!pip install networkx node2vec -q
import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt

# CELDA 2: Crear red de ejemplo (legisladores → comisiones)
G = nx.Graph()
legisladores = ['Ana', 'Bruno', 'Carla', 'Diego', 'Elena']
comisiones = ['Economía', 'Salud', 'Educación', 'Defensa']
for l in legisladores:
    for c in np.random.choice(comisiones, 2, replace=False):
        G.add_edge(l, c)

# CELDA 3: Node2Vec embeddings
node2vec = Node2Vec(G, dimensions=16, walk_length=5, num_walks=50)
model = node2vec.fit(window=3, min_count=1)

# CELDA 4: Encontrar legisladores similares
target = 'Ana'
similar = model.wv.most_similar(target)
print(f"👥 Legisladores similares a {target}:")
for name, score in similar:
    print(f"  {name}: {score:.3f}")

# CELDA 5: Visualizar red
nx.draw(G, with_labels=True, node_color='lightblue', font_size=10)
plt.title("Red Legisladores-Comisiones")
plt.show()
```

**Tiempo:** 1 min | **GPU:** NO | **Dataset:** CSV de relaciones

---

## 6. ScispaCy — NLP Científico/Legal

```python
# CELDA 1
!pip install scispacy -q
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz -q

import spacy
nlp = spacy.load("en_core_sci_sm")

# CELDA 2: Extraer entidades de texto científico
texto = """
The study analyzed 500 patients with Type 2 Diabetes Mellitus treated with Metformin 500mg. 
Results showed HbA1c reduction from 8.2% to 6.9% over 12 weeks. 
Dr. Sarah Chen at Massachusetts General Hospital led the research funded by NIH grant R01-DK123456.
"""
doc = nlp(texto)

print("🔬 Entidades extraídas:")
for ent in doc.ents:
    print(f"  {ent.text} → {ent.label_}")

print("\n✅ ScispaCy: NLP para textos biomédicos/científicos")
```

**Tiempo:** 2 min | **GPU:** NO | **Dataset:** Textos científicos

---

## 7. PySal — Análisis Geoespacial

```python
# CELDA 1
!pip install pysal esda geopandas libpysal -q
import numpy as np
import geopandas as gpd
from esda.moran import Moran
from libpysal.weights import Queen

# CELDA 2: Datos sintéticos de distritos
np.random.seed(42)
n = 30
coords = np.random.uniform(0, 10, (n, 2))
values = 50 + 20 * np.sin(coords[:, 0]) + np.random.normal(0, 5, n)

# CELDA 3: Autocorrelación espacial (Moran's I)
from libpysal.weights import DistanceBand
w = DistanceBand(coords, threshold=3.0)
moran = Moran(values, w)

print(f"📍 Moran's I: {moran.I:.3f} (p={moran.p_sim:.3f})")
print(f"   > 0 = clustering espacial, < 0 = dispersión")
print("✅ PySal: análisis de crímenes, acceso a salud, segregación")
```

**Tiempo:** 1 min | **GPU:** NO | **Dataset:** Datos geoespaciales

---

## 8. Quanteda (vía Python) — Análisis de Textos Legales

```python
# CELDA 1
!pip install nltk wordcloud matplotlib -q
import nltk
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# CELDA 2: "Leyes" peruanas de ejemplo
leyes = [
    "Ley de protección de datos personales establece que toda persona tiene derecho a la autodeterminación informativa y protección de sus datos personales en territorio peruano.",
    "Decreto de urgencia sobre reactivación económica aprueba medidas extraordinarias para la inversión pública y privada con enfoque en infraestructura digital.",
    "Ley de inteligencia artificial promueve el uso ético y responsable de la inteligencia artificial en el sector público y privado del Perú."
]

# CELDA 3: Análisis de frecuencia
stop_es = set(stopwords.words('spanish'))
all_words = []
for ley in leyes:
    tokens = nltk.word_tokenize(ley.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_es and len(t) > 3]
    all_words.extend(tokens)

freq = Counter(all_words).most_common(15)
print("📊 Palabras más frecuentes en leyes:")
for word, count in freq:
    print(f"  {word}: {count}")

# CELDA 4: Nube de palabras
wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(freq))
plt.imshow(wc)
plt.axis('off')
plt.title("Nube de Palabras — Leyes Peruanas")
plt.show()
```

**Tiempo:** 1 min | **GPU:** NO | **Dataset:** Textos legales
