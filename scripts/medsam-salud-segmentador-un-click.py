# ============================================================
# open-ai-toolkit | Segmentador por Clic (MedSAM)
# Rubro: Salud
# Para qué: Hacés clic en un órgano/tumor y lo segmenta automático
# Requiere: Imagen médica + GPU
# ============================================================

!pip install git+https://github.com/bowang-lab/MedSAM.git -q

print("✅ MedSAM: Segment Anything fine-tuneado para medicina")
print("🎯 1.5M+ máscaras de 11 modalidades médicas")
print("🖱️ Dibujá un rectángulo → MedSAM segmenta el resto")
print("📎 Demo: github.com/bowang-lab/MedSAM")
