# ============================================================
# open-ai-toolkit | Probador de Modelos ML Agrícolas (AgML)
# Rubro: AgriTech
# Para qué: Probá varios modelos y elegí el mejor para tu cultivo
# Demo: datasets incluidos | Datos propios: fotos etiquetadas
# ============================================================

!pip install agml -q
import agml

print("📊 Datasets agrícolas disponibles:")
for ds in agml.public_data_sources():
    print(f"  - {ds}")

loader = agml.data.AgMLDataLoader('apple_detection_segmentation')
loader.summary()
print("\n✅ Listo. Cargá tu dataset y compará modelos.")
