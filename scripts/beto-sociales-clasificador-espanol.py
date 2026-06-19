# ============================================================
# open-ai-toolkit | Clasificador de Textos en Español (BETO)
# Rubro: Ciencias Sociales
# Para qué: Leyes/tweets/notas → clasifica por tema en español
# Demo: inmediata | Datos propios: 100-500 textos etiquetados
# ============================================================

!pip install transformers torch -q
from transformers import pipeline

model = "dccuchile/bert-base-spanish-wwm-uncased"
classifier = pipeline('text-classification', model=model)

textos = [
    "El proyecto de ley propone modificar el artículo 14 de la constitución.",
    "La inflación bajó al 2.5% este mes según el BCRP.",
    "Se reportaron protestas en el sur por el alza de combustibles."
]
for t in textos:
    print(f"📝 {t[:70]}...")
    print(f"   → {classifier(t)[0]}")
print("\n✅ BETO: BERT en español (U. de Chile)")
