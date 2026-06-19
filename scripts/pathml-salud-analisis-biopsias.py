# ============================================================
# open-ai-toolkit | Analizador de Biopsias Digitales (PathML)
# Rubro: Salud / Patología
# Para qué: Lámina digitalizada → detecta células cancerígenas
# Requiere: Whole Slide Images (.svs, .tiff) + GPU
# ============================================================

!pip install pathml -q

print("✅ PathML instalado")
print("🔬 Análisis de WSI (Whole Slide Images)")
print("📊 Carga .svs → preprocesa → aplica HoverNet → cuantifica")
print("📎 Tutorial: github.com/Dana-Farber-AIOS/pathml")
