# ============================================================
# open-ai-toolkit | Asistente Legal Offline (GPT4All)
# Rubro: Ciencias Sociales
# Para qué: Documentos legales → resumen y análisis SIN internet
# Demo: inmediata (4GB RAM) | Datos propios: pegar textos
# ============================================================

!pip install gpt4all -q
from gpt4all import GPT4All

model = GPT4All("Meta-Llama-3.1-8B-Instruct-128k-GGUF")

respuesta = model.generate(
    "Resumí en 3 puntos este decreto:\n"
    "'Decreto Supremo N° 004-2024: medidas para transformación digital "
    "del Estado, interoperabilidad de sistemas, identidad digital única "
    "y capacitación de servidores públicos en competencias digitales.'",
    max_tokens=200
)
print(respuesta)
print("\n✅ GPT4All: LLM OFFLINE. Sin API. Sin internet. Datos seguros.")
