# ============================================================
# open-ai-toolkit | Explorador de Materiales (PyMatGen)
# Rubro: Nanotecnología y Materiales
# Para qué: Buscar entre 150k materiales por propiedad
# Demo: inmediata (sin GPU) | API key gratis: materialsproject.org
# ============================================================

!pip install pymatgen -q
from pymatgen.ext.matproj import MPRester

API_KEY = "TU_API_KEY_GRATIS"  # ← Registrate en materialsproject.org
with MPRester(API_KEY) as mpr:
    results = mpr.summary.search(
        chemsys="Li-O",
        fields=["material_id","formula_pretty","band_gap","energy_above_hull"],
        num_results=10
    )
    for r in results:
        print(f"🔬 {r.formula_pretty} | Band gap: {r.band_gap:.2f} eV | Stable: {r.energy_above_hull:.3f}")
print("\n📊 150,000+ materiales. API gratis.")
