# ============================================================
# open-ai-toolkit | Segmentación 3D de Tejidos Vegetales (PlantSeg)
# Rubro: AgriTech | Nicho: Biotecnología
# Para qué: Microscopio 3D → segmenta capas del tejido
# Requiere: Imagen .tiff/.h5 de microscopio 3D + GPU
# ============================================================

!pip install plantseg -q
import plantseg

print("✅ PlantSeg instalado")
print("🔬 Requiere imagen 3D de microscopio (.tiff o .h5)")
print("   Uso: run_plantseg('tu_imagen.tiff', model_name='generic_light_sheet_3d_unet')")
print("📎 Tutorial: github.com/hci-unihd/plant-seg")
