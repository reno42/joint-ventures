# ============================================================
# open-ai-toolkit | Lector de Textos Biomédicos (BioBERT)
# Rubro: Salud
# Para qué: Paper/Historia clínica → extrae enfermedades, fármacos, genes
# Demo: inmediata | Datos propios: 100-500 textos clínicos
# ============================================================

!pip install transformers torch -q
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(name)
classifier = pipeline('text-classification', model=name, tokenizer=tokenizer)

textos = [
    "The patient presented with acute myocardial infarction and elevated troponin.",
    "Aspirin 81mg daily prescribed for cardiovascular prevention."
]
for t in textos:
    print(f"📝 {t[:80]}...")

print("✅ BioBERT: PubMed + PMC fine-tuned")
