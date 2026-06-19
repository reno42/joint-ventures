# Modelos Open Source por Nicho — Bootcamp IA × Ciencia

## 🌾 AgriTech (INTA, agrónomos, biólogos de cultivos)

| Modelo | Qué hace | Licencia | Dónde corre |
|--------|----------|----------|-------------|
| **PlantVillage (PlantDoc)** | Detección de enfermedades en hojas por imagen (38 clases, 50K+ imágenes) | Open source | PyTorch, TensorFlow |
| **YOLOv8/v9 (Ultralytics)** | Detección de plagas en tiempo real con drones/cámaras | AGPL-3.0 | Edge devices, GPU |
| **SAM 2 (Meta - Segment Anything)** | Segmentación de cultivos, malezas, suelo desde imágenes satelitales | Apache 2.0 | GPU cloud |
| **Pllama** | LLM especializado en ciencia de plantas | Open | Ollama, vLLM |
| **AgriDoctor** | Asistente multimodal para diagnóstico agrícola (imagen + texto) | Research | API local |
| **TimesNet / PatchTST** | Predicción de rendimiento de cultivos con series temporales climáticas | MIT | PyTorch |
| **ClimaX (Microsoft)** | Modelo foundation para predicción climática | MIT | GPU |
| **Prithvi (NASA/IBM)** | Modelo foundation de teledetección (satélite → análisis de suelo/cultivo) | Apache 2.0 | Hugging Face |
| **DeepLabV3+** | Segmentación semántica de imágenes de campo | MIT | TensorFlow/PyTorch |
| **Random Forest / XGBoost** | Recomendación de dosis de fertilizantes, predicción de calidad de grano | BSD | Scikit-learn |

**Caso de uso típico:** "Tengo fotos de hojas de soja de un drone. Quiero detectar roya automáticamente y enviar alertas al productor."

---

## 🏥 Salud / BioTech (CONICET, bioquímicos, médicos)

| Modelo | Qué hace | Licencia | Dónde corre |
|--------|----------|----------|-------------|
| **Retinar (CONICET/UNICEN)** | Screening retinal con IA para prevenir ceguera (95.5% sensibilidad) | Propio (spin-off) | API, HSI |
| **MONAI (NVIDIA)** | Framework de deep learning para imágenes médicas 3D | Apache 2.0 | PyTorch |
| **MedSAM** | Segmentación de órganos/tumores en imágenes médicas | Apache 2.0 | GPU |
| **BioGPT (Microsoft)** | LLM para biomedicina (generación de texto, extracción de relaciones) | MIT | Ollama |
| **PubMedBERT** | NLP para literatura biomédica | MIT | Hugging Face |
| **AlphaFold 2 (DeepMind)** | Predicción de estructura de proteínas | Apache 2.0 | Colab, local GPU |
| **ESMFold (Meta)** | Predicción de estructura de proteínas (más rápido que AlphaFold) | MIT | GPU |
| **ChemBERTa** | Modelado molecular para descubrimiento de fármacos | MIT | Hugging Face |
| **DiffDock** | Docking molecular con diffusion models | MIT | GPU |
| **M3D** | Imágenes médicas 3D (avanzado) | Research | GPU |
| **CheXzero / CXR-LLaVA** | Interpretación de radiografías de tórax | MIT | API local |
| **LLaVA-Med** | Asistente multimodal para imágenes médicas | MIT | Ollama |

**Caso de uso típico:** "Tengo 10,000 retinografías del hospital público. Quiero pre-screening automático para derivar solo los casos de riesgo al oftalmólogo."

---

## 🧬 Nano / Materiales (FAN, nanotecnólogos)

| Modelo | Qué hace | Licencia | Dónde corre |
|--------|----------|----------|-------------|
| **MACE** | Potenciales de interacción atómica (simulación molecular) | MIT | Python |
| **CHGNet** | Modelo universal de energía/estructura cristalina | MIT | PyTorch |
| **M3GNet** | Dinámica molecular con ML (predicción de propiedades) | MIT | GPU |
| **CGCNN** | Red neuronal para propiedades de cristales | MIT | PyTorch |
| **ALIGNN** | Predicción de propiedades de materiales | MIT | PyTorch |
| **DeepMind GNoME** | Descubrimiento de nuevos materiales estables | Research | API |
| **ORB (Meta)** | Modelo universal para química computacional | MIT | GPU |
| **MolFormer** | Generación de moléculas con propiedades deseadas | MIT | Hugging Face |
| **OCP (Open Catalyst)** | Catálisis computacional (Meta/CMU) | MIT | PyTorch |

**Caso de uso típico:** "Quiero predecir qué combinaciones de elementos pueden crear un nanomaterial con conductividad específica, sin sintetizar miles de muestras."

---

## 🌍 Clima / Energía (climatólogos, ingenieros ambientales)

| Modelo | Qué hace | Licencia | Dónde corre |
|--------|----------|----------|-------------|
| **ClimaX (Microsoft)** | Modelo foundation para predicción climática | MIT | GPU |
| **Pangu-Weather (Huawei)** | Predicción meteorológica con transformer | Research | GPU |
| **GraphCast (Google)** | Predicción climática 10 días (supera modelos físicos) | Apache 2.0 | GPU |
| **Prithvi (NASA/IBM)** | Análisis de teledetección (incendios, inundaciones, sequías) | Apache 2.0 | Hugging Face |
| **FourCastNet (NVIDIA)** | Predicción climática de alta resolución | Apache 2.0 | GPU |
| **WattNet** | Optimización de redes eléctricas con ML | Research | PyTorch |
| **DeepSolar** | Detección de paneles solares desde imágenes satelitales | MIT | GPU |
| **RenewableNinja** | Predicción de generación solar/eólica | Open | API |

**Caso de uso típico:** "Quiero predecir la producción de energía eólica en Patagonia las próximas 72 horas para optimizar la distribución en la red."

---

## 📊 Ciencias Sociales (economistas, sociólogos, politólogos)

| Modelo | Qué hace | Licencia | Dónde corre |
|--------|----------|----------|-------------|
| **Llama 3.1 (Meta)** | LLM general, fine-tunable para análisis de políticas | Llama 3.1 License | Ollama, vLLM |
| **Mistral / Mixtral** | LLM eficiente, buen español | Apache 2.0 | Ollama |
| **Qwen 2.5 (Alibaba)** | LLM con buen soporte multilingüe | Apache 2.0 | Ollama |
| **BERT multilingual** | Clasificación de texto, análisis de sentimiento | Apache 2.0 | Hugging Face |
| **SBERT (Sentence-BERT)** | Embeddings semánticos para documentos largos | Apache 2.0 | Python |
| **Whisper (OpenAI)** | Transcripción de audio (audiencias, entrevistas, sesiones) | MIT | Local, API |
| **Phi-3 (Microsoft)** | LLM pequeño, eficiente, corre en laptops | MIT | Ollama |
| **Falcon** | LLM para generación de texto en español | Apache 2.0 | Ollama |
| **DeBERTa** | NLP avanzado para extracción de información | MIT | Hugging Face |

**Caso de uso típico:** "Tengo 500 documentos de políticas públicas provinciales. Quiero extraer automáticamente menciones de presupuesto, beneficiarios y resultados."

---

## 🔧 Herramientas transversales (todos los nichos)

| Herramienta | Qué hace | Para quién |
|-------------|----------|------------|
| **Ollama** | Correr modelos LLM locales en 1 línea de comando | Todos |
| **Hugging Face Hub** | Repositorio de 500K+ modelos pre-entrenados | Todos |
| **Gradio** | Crear UI web para modelos ML en minutos | Todos |
| **Streamlit** | Apps de datos interactivas | Todos |
| **FastAPI** | APIs de modelos ML | Todos |
| **LangChain** | Orquestar LLMs con herramientas externas | Todos |
| **vLLM** | Servir LLMs en producción (alto throughput) | Todos |
| **MLflow** | Tracking de experimentos ML | Todos |
| **DVC** | Versionado de datasets y modelos | Todos |
| **Label Studio** | Anotación de datos (imágenes, texto, audio) | Todos |

---

## 🎯 Mapeo modelo → nicho → módulo del bootcamp

| Módulo del Bootcamp | AgriTech | Salud | Nano | Clima | Sociales |
|---------------------|----------|-------|------|-------|----------|
| **M1: Problema de mercado** | Caso INTA drone | Caso Retinar | Caso FAN materiales | Caso energía eólica | Caso políticas públicas |
| **M2: Herramientas IA** | YOLO + SAM + PlantVillage | MONAI + MedSAM | MACE + CHGNet | ClimaX + Prithvi | Llama + Whisper |
| **M3: Prototipado** | App detección plagas | App screening retinal | Simulador materiales | Predictor climático | Analizador documentos |
| **M4: Modelo de negocio** | SaaS para productores | Telemedicina API | Software para laboratorios | Dashboard para distribuidoras | Herramienta para think tanks |
| **M5: IP y patentes** | Patente algoritmo detección | Patente método diagnóstico | Patente composición | Patente método predicción | Copyright del modelo |

---

## 💡 Stack mínimo para el bootcamp

```bash
# Lo que cada participante necesita instalar:
- Python 3.11+
- Ollama (para LLMs locales)
- Hugging Face account (para modelos)
- Google Colab (GPU gratuita)
- VS Code + GitHub

# Modelos base recomendados (arrancan acá):
- Llama 3.1 8B (vía Ollama) — LLM general
- SAM 2 — Segmentación de imágenes
- Whisper — Transcripción de audio
- YOLOv8 — Detección de objetos
- BERT multilingual — NLP en español

# Luego, según nicho:
- AgriTech → PlantVillage + Prithvi
- Salud → MONAI + MedSAM
- Nano → MACE + CHGNet
- Clima → ClimaX + Prithvi
- Sociales → Llama fine-tuned + SBERT
```
