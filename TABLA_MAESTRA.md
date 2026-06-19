# 📊 Master Table — 37 Herramientas Open Source por Rubro

> Para investigadores y emprendedores en LATAM. Todo ejecutable en Google Colab.

| # | Rubro | Modelo | ¿Para qué sirve? | ¿Trae dataset de ejemplo? | ¿Necesito algo extra? | ¿Script para mis datos? |
|---|-------|--------|-----------------|:------------------------:|:---------------------:|:-----------------------:|
| 1 | 🌾 AgriTech | **YOLOv8** | Detectar plagas, frutos, objetos en fotos | ✅ Incluye imagen de prueba (bus) | ❌ Nada | ✅ Subir foto: `files.upload()` incluido |
| 2 | 🌾 AgriTech | **PlantVillage** | Dataset de 38 enfermedades en 14 cultivos (54k imágenes) | ✅ 54,000 imágenes etiquetadas | ❌ Nada | ⚠️ Hay que entrenar YOLOv8 con este dataset |
| 3 | 🌾 AgriTech | **Crop Yield** | Predecir rendimiento de cosecha con datos de clima | ✅ Datos sintéticos de ejemplo | ❌ Nada | ⚠️ Reemplazar CSV sintético por datos reales |
| 4 | 🌾 AgriTech | **PlantSeg** | Segmentación 3D de tejidos vegetales (microscopio) | ❌ Requiere imágenes 3D .tiff | 🔴 Colab Pro (GPU) | ⚠️ Necesitás tus propias imágenes de microscopio |
| 5 | 🌾 AgriTech | **AgML** | Comparar qué modelo de ML funciona mejor para tu cultivo | ✅ Varios datasets incluidos | ❌ Nada | ✅ Datasets de ejemplo incluidos |
| 6 | 🌾 AgriTech | **DeepWeeds** | Clasificar malezas con ResNet | ✅ 17,509 imágenes de 8 malezas | ❌ Nada | ✅ Dataset incluido, o subís tus fotos |
| 7 | 🌾 AgriTech | **Irrigación** | Optimizar cuándo y cuánto regar | ✅ Datos sintéticos de ejemplo | ❌ Nada | ⚠️ Reemplazar CSV por datos de sensores IoT |
| 8 | 🏥 Salud | **MONAI** | Segmentación de tumores/órganos en imágenes médicas | ❌ Requiere DICOM/NIfTI | 🔴 Colab Pro (GPU) | ⚠️ Necesitás imágenes médicas de tu hospital |
| 9 | 🏥 Salud | **CheXNet** | Detectar neumonía en rayos-X de tórax | ✅ NIH ChestX-ray14 (112k imágenes) | ❌ Nada | ✅ Subir radiografía: código incluido |
| 10 | 🏥 Salud | **nnU-Net** | Segmentación médica auto-configurable | ❌ Requiere imágenes + máscaras | 🔴 Colab Pro (GPU) | ⚠️ Preparar datos en formato nnU-Net |
| 11 | 🏥 Salud | **BioBERT** | Extraer info de textos biomédicos (PubMed, historias clínicas) | ✅ Textos de ejemplo incluidos | ❌ Nada | ✅ Pegar tus textos: código incluido |
| 12 | 🏥 Salud | **DeepLabCut** | Tracking de movimiento en videos (personas/animales) | ❌ Requiere tus propios videos | 🟡 GPU recomendada | ⚠️ Etiquetar 20-50 frames manualmente |
| 13 | 🏥 Salud | **AlphaFold** | Predecir estructura 3D de proteínas (Premio Nobel 2024) | ✅ Secuencia de ejemplo incluida | 🟡 GPU T4 (Colab Free OK) | ✅ Pegar tu secuencia de aminoácidos |
| 14 | 🏥 Salud | **PathML** | Análisis de biopsias digitalizadas (cáncer) | ❌ Requiere WSI (.svs, .tiff) | 🔴 Colab Pro (GPU) | ⚠️ Necesitás láminas digitalizadas de patología |
| 15 | 🏥 Salud | **MedSAM** | Segmentar órganos/tumores con un clic (Segment Anything) | ❌ Requiere imagen médica + bounding box | 🔴 Colab Pro (GPU) | ⚠️ Dibujar bounding box en tu imagen médica |
| 16 | ⚛️ Materiales | **PyMatGen** | Explorar 150k+ materiales (baterías, semiconductores) | ✅ API gratis del Materials Project | ❌ Nada (sin GPU) | ✅ Buscar materiales por fórmula: código incluido |
| 17 | ⚛️ Materiales | **MatGL** | Predecir propiedades de materiales con GNNs | ✅ Estructura Li2O de ejemplo | 🟡 GPU recomendada | ✅ Crear tu estructura cristalina en código |
| 18 | ⚛️ Materiales | **DeepMD-kit** | Simulaciones atómicas 1000x más rápido que DFT | ❌ Requiere datos de DFT (VASP/QE) | 🔴 Colab Pro (GPU) | ⚠️ Generar datos con VASP o Quantum ESPRESSO |
| 19 | ⚛️ Materiales | **MACE** | Modelos equivariantes (next-gen, más rápido que DeepMD) | ✅ Molécula H2O de ejemplo | 🟡 GPU opcional | ✅ Crear tus átomos en código |

| 20 | ⚛️ Materiales | **CHGNet** | Diseño de cátodos para baterías de litio | ✅ Estructura de ejemplo | 🟡 GPU recomendada | ✅ Crear estructura de batería en código |
| 21 | ⚛️ Materiales | **SchNetPack** | Predecir energía y fuerzas de moléculas | ✅ Dataset QM9 incluido (134k moléculas) | 🟡 GPU recomendada | ✅ Entrenar con QM9 o tus propias moléculas |
| 22 | ⚛️ Materiales | **DGL-LifeSci** | Drug discovery: predecir toxicidad, solubilidad | ✅ Dataset MoleculeNet incluido | 🟡 GPU recomendada | ✅ Datasets MoleculeNet o tus compuestos |
| 23 | 🌍 Clima | **Prophet** | Forecasting de temperatura, ventas, demanda | ✅ Datos sintéticos de ejemplo | ❌ Nada (sin GPU) | ✅ Reemplazar CSV por tus datos históricos |
| 24 | 🌍 Clima | **PyTorch Forecasting** | Forecasting multivariable con Transformers | ✅ Datos sintéticos de ejemplo | 🟡 GPU opcional | ⚠️ Preparar CSV multivariable con timestamp + features |
| 25 | 🌍 Clima | **OpenClimateFix** | Predecir producción de energía solar próximas 4h | ❌ Requiere datos satelitales + parques solares | 🟡 GPU recomendada | ⚠️ Necesitás datos de satélite + producción real |
| 26 | 🌍 Clima | **ClimateLearn** | Forecasting + downscaling climático regional | ❌ Requiere dataset ERA5 (~50GB) | 🟡 GPU recomendada | ⚠️ Descargar ERA5 del CDS Copernicus (gratis) |
| 27 | 🌍 Clima | **FourCastNet** | Predicción climática global (modelo NVIDIA) | ❌ Requiere dataset ERA5 (~50GB) | 🔴 GPU A100 recomendada | ⚠️ Dataset ERA5 completo + pesos pre-entrenados |
| 28 | 🌍 Clima | **DeepSensor** | Interpolar datos entre estaciones meteorológicas | ✅ Datos sintéticos de ejemplo | ❌ Nada | ✅ Reemplazar coordenadas + mediciones reales |
| 29 | 🌍 Clima | **Wind Power** | Predecir producción de energía eólica | ✅ Datos sintéticos de viento | ❌ Nada | ✅ Reemplazar CSV por velocidad viento + producción |
| 30 | 📊 C. Sociales | **BETO** | Clasificar textos en español (leyes, tweets) | ✅ Textos de ejemplo en español | ❌ Nada | ✅ Pegar tus textos: clasificación inmediata |
| 31 | 📊 C. Sociales | **MarIA** | NLP avanzado en español (modelo del BSC Barcelona) | ✅ Fill-mask de ejemplo | 🟡 GPU recomendada | ✅ Pegar tus textos: modelo más potente que BETO |
| 32 | 📊 C. Sociales | **GPT4All** | LLM offline: analizar documentos sin API ni internet | ✅ Prompt de ejemplo incluido | ❌ Nada (4GB RAM) | ✅ Pegar tu texto legal: análisis inmediato |
| 33 | 📊 C. Sociales | **BERTopic** | Descubrir temas automáticamente en cientos de documentos | ✅ 20 Newsgroups de ejemplo | ❌ Nada | ✅ Reemplazar por tu corpus de leyes/decretos |
| 34 | 📊 C. Sociales | **NetworkX + Node2Vec** | Analizar redes: quién influye en quién | ✅ Red sintética de ejemplo | ❌ Nada | ✅ Reemplazar CSV de relaciones (nombre, conexión) |
| 35 | 📊 C. Sociales | **ScispaCy** | Extraer entidades de textos científicos/legales | ✅ Texto científico de ejemplo | ❌ Nada | ✅ Pegar tu paper o documento legal |
| 36 | 📊 C. Sociales | **PySal** | Análisis geoespacial: clustering, segregación | ✅ Datos sintéticos de ejemplo | ❌ Nada | ✅ Reemplazar coordenadas + indicadores reales |
| 37 | 📊 C. Sociales | **Quanteda/NLTK** | Análisis de frecuencia y nubes de palabras en leyes | ✅ Leyes peruanas de ejemplo | ❌ Nada | ✅ Pegar tus decretos o discursos |

---

## 📊 RESUMEN POR TIPO DE SETUP

| Tipo | Cantidad | Herramientas |
|------|:--------:|-------------|
| ✅ **YA trae dataset, ejecuta sin configurar nada** | 22 | YOLOv8, PlantVillage, Crop Yield, AgML, DeepWeeds, Irrigación, CheXNet, BioBERT, AlphaFold, PyMatGen, MatGL, MACE, CHGNet, SchNetPack, DGL-LifeSci, Prophet, PyT Forecasting, DeepSensor, Wind Power, BETO, MarIA, GPT4All, BERTopic, NetworkX, ScispaCy, PySal, Quanteda |
| ⚠️ **Necesita tus datos (script incluido para cargarlos)** | 10 | Crop Yield, Irrigación, BioBERT, AlphaFold, PyMatGen, MACE, Prophet, DeepSensor, Wind Power, BETO, GPT4All, BERTopic, NetworkX, Quanteda |
| 🔴 **Requiere Colab Pro + datos especializados** | 5 | PlantSeg, MONAI, nnU-Net, PathML, MedSAM, DeepMD-kit, FourCastNet |

---

## 🎯 LAS 5 MÁS FÁCILES PARA ARRANCAR HOY

| # | Herramienta | Rubro | Setup | Tiempo hasta resultado |
|---|------------|-------|-------|:----------------------:|
| 1 | **YOLOv8** | AgriTech | 1 `pip install` | 2 min |
| 2 | **Prophet** | Clima | 1 `pip install` + CSV | 1 min |
| 3 | **BETO** | C. Sociales | Descarga modelo 400MB | 2 min |
| 4 | **GPT4All** | C. Sociales | Descarga modelo 4GB | 5 min |
| 5 | **CheXNet** | Salud | Modelo pre-entrenado | 3 min |
