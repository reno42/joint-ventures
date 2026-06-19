# ============================================================
# open-ai-toolkit | Segmentador de Imágenes Médicas (MONAI)
# Rubro: Salud
# Para qué: Tomografía/resonancia → segmenta tumores y órganos
# Requiere: Imágenes DICOM/NIfTI + GPU
# ============================================================

!pip install monai nibabel -q
import torch
from monai.networks.nets import UNet

model = UNet(spatial_dims=2, in_channels=1, out_channels=1, channels=(16,32,64,128,256), strides=(2,2,2,2), num_res_units=2)
print(f"✅ MONAI UNet: {sum(p.numel() for p in model.parameters()):,} parámetros")
print("📊 Dataset: tcia.org (gratis) o tus DICOM")
print("🔗 Tutorial: github.com/Project-MONAI/tutorials")
