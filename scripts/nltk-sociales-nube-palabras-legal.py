# ============================================================
# open-ai-toolkit | Nube de Palabras Legal (NLTK)
# Rubro: Ciencias Sociales
# Para qué: Discurso/ley/decreto → palabras más usadas + nube visual
# Demo: leyes peruanas ejemplo | Datos propios: 1 archivo TXT
# ============================================================

!pip install nltk wordcloud matplotlib -q
import nltk; nltk.download(['stopwords','punkt','punkt_tab'])
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

leyes = [
    "Ley de protección de datos personales establece que toda persona tiene derecho a la autodeterminación informativa.",
    "Decreto de urgencia sobre reactivación económica aprueba medidas extraordinarias para inversión pública y privada.",
    "Ley de inteligencia artificial promueve el uso ético y responsable de IA en el sector público y privado del Perú."
]

stop = set(nltk.corpus.stopwords.words('spanish'))
words = []
for ley in leyes:
    tokens = nltk.word_tokenize(ley.lower())
    words.extend([t for t in tokens if t.isalpha() and t not in stop and len(t)>3])

freq = Counter(words).most_common(15)
print("📊 Palabras más frecuentes:")
for w, c in freq: print(f"  {w}: {c}")

wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(freq))
plt.imshow(wc); plt.axis('off'); plt.title("Nube de Palabras"); plt.show()
