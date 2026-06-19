# ============================================================
# open-ai-toolkit | Analizador de Movimiento (DeepLabCut)
# Rubro: Salud / Neurociencia
# Para qué: Video de paciente → trackea articulaciones sin marcadores
# Requiere: Videos MP4 + etiquetar 20-50 frames
# ============================================================

!pip install deeplabcut -q
import deeplabcut

print("✅ DeepLabCut instalado")
print("🎥 Flujo: 1) Crear proyecto  2) Extraer frames  3) Etiquetar")
print("         4) Entrenar red  5) Analizar videos automático")
print("🧠 Usos: fisioterapia, deportes, comportamiento animal")
