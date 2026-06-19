# ============================================================
# open-ai-toolkit | Segmentador Médico Automático (nnU-Net)
# Rubro: Salud
# Para qué: Se auto-configura solo — analiza tus imágenes y elige la mejor red
# Requiere: Imágenes + máscaras + GPU
# ============================================================

!pip install nnunetv2 -q

print("✅ nnU-Net instalado — State-of-the-art en 33 benchmarks")
print("🔧 Uso: nnUNetv2_plan_and_preprocess → nnUNetv2_train → nnUNetv2_predict")
print("📎 Tutorial: github.com/MIC-DKFZ/nnUNet")
