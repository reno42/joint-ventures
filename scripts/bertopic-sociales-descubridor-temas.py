# ============================================================
# open-ai-toolkit | Descubridor de Temas (BERTopic)
# Rubro: Ciencias Sociales
# Para qué: 500 documentos → encuentra temas automáticamente
# Demo: inmediata | Datos propios: corpus de textos (100-500 docs)
# ============================================================

!pip install bertopic -q
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))['data'][:500]

model = BERTopic(language="multilingual", verbose=True)
topics, _ = model.fit_transform(docs)

info = model.get_topic_info()
print(f"\n📊 {len(info)} temas descubiertos:")
for i, row in info.head(8).iterrows():
    if row['Topic'] != -1:
        words = model.get_topic(row['Topic'])
        print(f"  Tema {row['Topic']}: {', '.join([w[0] for w in words[:5]])} ({row['Count']} docs)")

print("\n✅ BERTopic: temas emergentes sin supervisión")
