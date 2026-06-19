# ============================================================
# open-ai-toolkit | Analizador Avanzado de Español (MarIA)
# Rubro: Ciencias Sociales
# Para qué: Más potente que BETO — Barcelona Supercomputing Center
# Demo: inmediata | Datos propios: 100-500 textos
# ============================================================

!pip install transformers -q
from transformers import pipeline

model = "PlanTL-GOB-ES/roberta-base-bne"
fill = pipeline('fill-mask', model=model)

result = fill("La nueva ley de protección de datos <mask> los derechos ciudadanos.")
for r in result:
    print(f"🔤 {r['token_str']}: {r['score']:.3f}")

print("✅ MarIA: RoBERTa español entrenado en la Biblioteca Nacional de España")
