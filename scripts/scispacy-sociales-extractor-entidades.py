# ============================================================
# open-ai-toolkit | Extractor de Entidades Científicas (ScispaCy)
# Rubro: Ciencias Sociales / Investigación
# Para qué: Paper → extrae enfermedades, genes, fármacos
# Demo: texto inglés incluido | Español: requiere fine-tune
# ============================================================

!pip install scispacy -q
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz -q
import spacy

nlp = spacy.load("en_core_sci_sm")
texto = "500 patients with Type 2 Diabetes treated with Metformin 500mg. HbA1c reduced from 8.2% to 6.9% over 12 weeks."
doc = nlp(texto)

print("🔬 Entidades extraídas:")
for ent in doc.ents:
    print(f"  {ent.text} → {ent.label_}")
