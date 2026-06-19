# ============================================================
# open-ai-toolkit | Predictor de Estructura de Proteínas (AlphaFold)
# Rubro: Salud / BioTech
# Para qué: Secuencia de aminoácidos → estructura 3D (Premio Nobel 2024)
# Demo: secuencia ejemplo | Datos propios: tu secuencia
# ============================================================

!pip install -q --no-warn-conflicts "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"

from colabfold.batch import run

query = ("proteina_ejemplo", "MHHHHHHSSGVDLGTENLYFQSNAGSETVRFLAYDGWSFLASGGLGGQEAIAQAVGQALDA")

print("🧬 AlphaFold — DeepMind (Premio Nobel Química 2024)")
print("⏳ Ejecutando predicción en GPU T4 (5-10 min)...")
print(f"📐 Secuencia: {query[1][:40]}...")

try:
    results = run(queries=[query], num_models=1, use_amber=False)
    print("✅ Estructura 3D lista. Archivos PDB generados.")
except Exception as e:
    print(f"⚠️ GPU limitada: {e}")
    print("📎 Usá el notebook oficial: colab.research.google.com/github/sokrypton/ColabFold")
