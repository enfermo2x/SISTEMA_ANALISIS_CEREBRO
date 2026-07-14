# PLAN DE EJECUCIÓN DEL PROYECTO
## Sistema Inteligente para la Detección de Tumores Cerebrales en MRI mediante CNN

---

## DOCUMENTO EJECUTIVO

Este documento contiene el análisis de los requerimientos del proyecto y el plan detallado de ejecución, dividido por roles y por módulos técnicos, para llevarlo a un resultado completo (100%).

**Grupo**: 3 integrantes
**Estado**: Fase 1 y Fase 2 completadas
**Próximo paso**: Inicio de Fase 3 (CNN propia)

---

## I. CONTEXTO Y DATASET

### Contexto
Un centro de diagnóstico por imágenes requiere una herramienta de apoyo que clasifique automáticamente resonancias magnéticas cerebrales, identificando si existe un tumor y de qué tipo, como segunda opinión rápida para personal médico. El sistema expone sus predicciones mediante una API y una interfaz web para cargar imágenes.

### Dataset
**Brain Tumor MRI Dataset** (Kaggle — Masoud Nickparvar), combinación de figshare, SARTAJ y Br35H. ~7,023 imágenes en escala de grises, ya divididas en Training/Testing.

| Clase | Descripción | Volumen aprox. |
|---|---|---|
| Glioma | Tumor en células gliales del cerebro/médula | ~1621 imágenes |
| Meningioma | Tumor en las meninges | ~1645 imágenes |
| Pituitary | Tumor en la glándula pituitaria | ~1757 imágenes |
| No Tumor | MRI sin evidencia de tumor | ~2000 imágenes |

Enlace: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

---

## II. DISTRIBUCIÓN DE ROLES Y FASES

### Rol 1 — Data Scientist / Deep Learning Lead (70%)

| Fase | Entregable | Peso |
|---|---|---|
| 1. Exploración del dataset | Conteo por clase, balance, resolución, calidad | 10% |
| 2. Preprocesamiento | Resize, normalización, escala de grises, split, Data Augmentation | 15% |
| 3. CNN propia | Red convolucional multiclase entrenada desde cero | 20% |
| 4. Transfer Learning | Comparación con ResNet50 (o MobileNetV2/EfficientNetB0) | 15% |
| 5. Evaluación | Accuracy, precision, recall, F1 por clase, matriz de confusión, curvas | 10% |

### Rol 2 — Backend / DevOps Engineer (20%)

| Fase | Entregable | Peso |
|---|---|---|
| 6. API + Despliegue | Flask con endpoint `/predict`, contenedorizado (Docker) o como servicio (systemd) | 20% |

### Rol 3 — Frontend Engineer / QA (10%)

| Fase | Entregable | Peso |
|---|---|---|
| 7. Frontend | Carga de imagen, consumo de `/predict`, visualización de resultado, QA end-to-end | 10% |

### Preguntas de análisis (transversal, sin peso individual asignado — responsabilidad conjunta al cierre)
9 preguntas conceptuales sobre CNN, Data Augmentation, desbalance de clases, overfitting, Softmax, recall en contexto médico, robustez, Docker vs systemd, y ética.

---

## III. ARQUITECTURA PROPUESTA DEL REPOSITORIO

```
brain-tumor-cnn/
├── Training/                        # Dataset original (training)
│   ├── glioma/                      # Tr-gl_*.jpg
│   ├── meningioma/                  # Tr-me_*.jpg, Tr-aug-me_*.jpg
│   ├── notumor/                     # Tr-no_*.jpg
│   └── pituitary/                   # Tr-pi_*.jpg
├── Testing/                         # Dataset original (testing)
│   ├── glioma/                      # Te-gl_*.jpg
│   ├── meningioma/                  # Te-me_*.jpg, Te-aug-me_*.jpg
│   ├── notumor/                     # Te-no_*.jpg
│   └── pituitary/                   # Te-pi_*.jpg
├── data/
│   └── processed/                   # Datos preprocesados
├── notebooks/                       # Exploración y prototipado
│   ├── 01_eda.ipynb
│   ├── 02_preprocesamiento.ipynb
│   ├── 03_cnn_propia.ipynb
│   ├── 04_transfer_learning.ipynb
│   └── 05_evaluacion.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                # Carga de imágenes
│   │   ├── preprocessing.py         # Resize, normalización, escala de grises
│   │   └── augmentation.py          # Data Augmentation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_custom.py            # Arquitectura CNN propia
│   │   ├── transfer_learning.py     # ResNet50 / MobileNetV2
│   │   └── train.py                 # Loop de entrenamiento
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Accuracy, precision, recall, F1
│   │   └── plots.py                 # Curvas y matriz de confusión
│   └── utils/
│       ├── __init__.py
│       └── config.py                # Configuración global (rutas, hiperparámetros)
├── models/                          # Modelos entrenados (.h5 / .pt)
│   ├── cnn_custom.h5
│   └── resnet50_finetuned.h5
├── api/
│   ├── app.py                       # Flask, endpoint /predict
│   ├── inference.py                 # Carga de modelo y predicción
│   ├── requirements.txt
│   ├── Dockerfile
│   └── braintumor-api.service       # Unidad systemd (opción B)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js                       # Consumo de /predict
├── tests/
│   ├── test_preprocessing.py
│   ├── test_api.py
│   └── test_frontend_e2e.py         # Pruebas QA end-to-end
├── docs/
│   ├── README.md
│   └── respuestas_analisis.md       # Respuestas a las 9 preguntas
├── requirements.txt
└── .gitignore
```

### Ventajas de esta arquitectura
✓ Separa claramente el trabajo de los 3 roles sin bloquear a nadie
✓ `models/` es el contrato entre Data Science y Backend (un `.h5`/`.pt` con formato conocido)
✓ `api/` expone un JSON fijo, que es el contrato entre Backend y Frontend
✓ Notebooks para exploración, `src/` para código reutilizable y testeable
✓ Despliegue documentado como comandos de consola, reproducible sin entorno gráfico

---

## IV. PLAN DE IMPLEMENTACIÓN POR MÓDULOS

### Módulo 1 — Exploración del dataset (Rol 1 · Fase 1)
**Archivos**: `notebooks/01_eda.ipynb`, `src/data/loader.py`
**Tareas**:
- Cargar y contar imágenes por clase (Training/Testing)
- Verificar balance entre las 4 clases
- Revisar resolución y calidad (imágenes corruptas, duplicados)
**Dependencias**: Ninguna
**Salida**: Reporte de EDA + gráficos de distribución

---

### Módulo 2 — Preprocesamiento y Augmentation (Rol 1 · Fase 2)
**Archivos**: `src/data/preprocessing.py`, `src/data/augmentation.py`
**Tareas**:
- Resize uniforme (ej. 224x224 o 150x150)
- Normalización de píxeles
- Split train/validation/test
- Data Augmentation (rotación, flip, zoom, brillo)
**Dependencias**: Módulo 1
**Salida**: Generadores/datasets listos para entrenar

---

### Módulo 3 — CNN propia (Rol 1 · Fase 3)
**Archivos**: `src/models/cnn_custom.py`, `notebooks/03_cnn_propia.ipynb`
**Tareas**:
- Diseñar arquitectura (Conv2D + Pooling + Dense + Softmax de 4 salidas)
- Entrenar sin pesos preentrenados
- Guardar checkpoints y curvas de entrenamiento
**Dependencias**: Módulo 2
**Salida**: `models/cnn_custom.h5`

---

### Módulo 4 — Transfer Learning (Rol 1 · Fase 4)
**Archivos**: `src/models/transfer_learning.py`
**Tareas**:
- Cargar ResNet50 (o MobileNetV2/EfficientNetB0) preentrenada
- Fine-tuning sobre las 4 clases
- Comparar contra la CNN propia
**Dependencias**: Módulo 2, 3
**Salida**: `models/resnet50_finetuned.h5` + tabla comparativa

---

### Módulo 5 — Evaluación (Rol 1 · Fase 5)
**Archivos**: `src/evaluation/metrics.py`, `src/evaluation/plots.py`
**Tareas**:
- Calcular accuracy, precision, recall y F1 por clase (foco en recall médico)
- Matriz de confusión
- Curvas de entrenamiento (loss/accuracy por época)
**Dependencias**: Módulo 3, 4
**Salida**: Reporte de métricas + gráficos finales, modelo elegido para producción

---

### Módulo 6 — API y despliegue (Rol 2 · Fase 6)
**Archivos**: `api/app.py`, `api/inference.py`, `api/Dockerfile` o `api/braintumor-api.service`
**Tareas**:
- Endpoint `POST /predict` (multipart/form-data, campo `image`)
- Cargar el modelo elegido en Módulo 5 y correr inferencia
- Responder JSON: `{"tipo_tumor": "...", "probabilidad": 97.3}`
- Elegir despliegue:
  - **Docker**: `Dockerfile` → `docker build` → `docker run -d -p 5000:5000 --name braintumor-api braintumor-api`
  - **systemd**: entorno virtual + `braintumor-api.service` → `systemctl enable/start braintumor-api`
- Documentar logs, reinicio y verificación (`docker logs` / `journalctl -u braintumor-api -f`)
**Dependencias**: Módulo 5 (modelo entrenado)
**Salida**: API corriendo y accesible en `localhost:5000/predict`

---

### Módulo 7 — Frontend y QA (Rol 3 · Fase 7)
**Archivos**: `frontend/index.html`, `frontend/app.js`, `tests/test_frontend_e2e.py`
**Tareas**:
- Formulario para cargar imagen MRI
- Consumo de `/predict` vía `fetch`
- Mostrar `tipo_tumor` y `probabilidad` en una tarjeta, con indicador visual distinto para "No Tumor"
- Aviso visible: el resultado no reemplaza diagnóstico médico
- Manejo de errores (imagen inválida, servidor caído, respuesta malformada)
- Pruebas end-to-end sobre el flujo completo (carga → predicción → visualización)
**Dependencias**: Módulo 6 (API disponible)
**Salida**: Interfaz funcional conectada a la API

---

### Módulo 8 — Respuestas de análisis y documentación (Transversal)
**Archivos**: `docs/respuestas_analisis.md`, `README.md`
**Tareas**:
- Responder las 9 preguntas de análisis con base en los resultados obtenidos
- Documentar instalación, uso y despliegue en el README
- Preparar el repositorio de GitHub final
**Dependencias**: Módulos 1-7
**Salida**: Repositorio listo para entrega

---

## V. CRONOGRAMA / CHECKLIST GENERAL

- [x] Módulo 1 — EDA
- [x] Módulo 2 — Preprocesamiento y Augmentation
- [ ] Módulo 3 — CNN propia
- [ ] Módulo 4 — Transfer Learning
- [ ] Módulo 5 — Evaluación y selección del modelo final
- [ ] Módulo 6 — API + despliegue (Docker o systemd)
- [ ] Módulo 7 — Frontend + QA end-to-end
- [ ] Módulo 8 — Respuestas de análisis + documentación + repo en GitHub

**Regla de oro**: Backend (Módulo 6) no puede empezar la integración real sin un modelo entregado por el Módulo 5. Frontend (Módulo 7) puede maquetar en paralelo, pero las pruebas E2E dependen de que la API esté corriendo.

---

## VI. RESUMEN EJECUTIVO

### Estado objetivo (100%)
✓ Dataset explorado y documentado
✓ CNN propia entrenada y evaluada
✓ Transfer Learning comparado contra la CNN propia
✓ Métricas completas, con foco en recall por el contexto médico
✓ API desplegada por consola (Docker o systemd), con logs y reinicio documentados
✓ Frontend funcional con manejo de errores y aviso médico visible
✓ Preguntas de análisis respondidas
✓ Repositorio en GitHub completo

### Distribución de peso final

| Rol | Fases | Peso total |
|---|---|---|
| Data Scientist / DL Lead | 1-5 | 70% |
| Backend / DevOps | 6 | 20% |
| Frontend / QA | 7 | 10% |

---

## VII. DIAGNÓSTICO Y MEJORAS — CNN PROPIA (Módulo 3)

### Problema detectado
La CNN propia original obtenía **accuracy=0.25 (25%)** en train y validation, equivalente a predicción aleatoria para 4 clases. El loss convergía a ~1.386 (ln(4)), indicando que el modelo aprendía a predecir siempre una misma clase.

### Causas raíz identificadas

| Causa | Detalle | Archivo |
|-------|---------|---------|
| **1 conv por bloque** | La red tenía solo 1 capa Conv2D por bloque, insuficiente para aprender features complejos en MRI. | `src/models/cnn_custom.py` |
| **Dropout agresivo (0.5)** | Dropout=0.5 en la primera Dense mataba el gradiente en etapas tempranas. | `src/models/cnn_custom.py` |
| **Inicialización genérica** | No se usaba `he_normal`, que es la inicialización recomendada para ReLU. | `src/models/cnn_custom.py` |
| **Augmentation como Sequential en tf.data** | El pipeline usaba un modelo `Sequential` dentro de `dataset.map()`, lo que puede causar issues de compatibilidad con TF 2.21. | `src/data/augmentation.py` |
| **Early stopping monitoreaba val_loss** | El early stopping original usaba `val_loss`, permitiendo que el modelo se estancara en val_accuracy=0.25 sin activarse. | `src/models/train.py` |

### Mejoras aplicadas (archivos modificados) — V2.1

| Archivo | Cambio |
|---------|--------|
| `src/utils/config.py` | `IMG_SIZE` = (128,128) (era 224x224 → 4x más rápido), `BATCH_SIZE` = 128 (era 32) |
| `src/data/loader.py` | **Carga en escala de grises (1 canal)** en vez de RGB, añade channel dim automáticamente |
| `src/models/cnn_custom.py` | 2 Conv2D por bloque + he_normal init + Dropout(0.3) + Dense(512) + **input_shape=(128,128,1)** |
| `src/models/train.py` | **CosineDecay LR schedule** en vez de ReduceLROnPlateau — LR baja suavemente desde el inicio |
| `src/data/augmentation.py` | Augmentation con `tf.image.*` directamente (sin Sequential), factores conservadores |
| `notebooks/03_cnn_propia.ipynb` | **Training** (5600) → train/val split, **Testing** (1600) → evaluación final; guarda como `.keras` |
| `notebooks/04_transfer_learning.ipynb` | Guarda como `.keras` en vez de `.h5` |
| `notebooks/05_evaluacion.ipynb` | Carga modelos `.keras` en vez de `.h5` |
| `tests/test_cnn_configs.py` | Script de testeo rápido con **8 arquitecturas**, entrena 1 epoch y evalúa si supera umbral |

---

## VIII. INSTRUCCIONES DE EJECUCIÓN

### Paso 1: Test rápido (1 epoch) para validar arquitectura

Esto ejecuta las 7 variantes de arquitectura con **30% de los datos y sin augmentation** para obtener resultados en ~5-10 minutos:

```powershell
python tests/test_cnn_configs.py --mode fast
```

**Criterio de validación:**
- **EXCELENTE** (val_acc ≥ 0.79): La arquitectura funciona. Proceder al Paso 2.
- **BUENO** (val_acc ≥ 0.50): Aprendizaje significativo. Proceder al Paso 2 igualmente.
- **PROMETEDOR** (val_acc ≥ 0.35): Hay aprendizaje. Se puede mejorar con más epochs.
- **MALO** (< 0.27): La arquitectura no aprende. Revisar.

Para testear una arquitectura específica:
```powershell
python tests/test_cnn_configs.py --mode fast --arch v7_improved_from_src
```

### Paso 2: Ejecutar el notebook completo

Ya se validó que la arquitectura funciona (val_acc 50% en época 1 desde 25% aleatorio).  
La configuración actual está optimizada para **128x128** y **batch_size=128**, cargando **7200 imágenes** (Training + Testing).

Ejecutar el notebook:
```powershell
jupyter notebook notebooks/03_cnn_propia.ipynb
```

O desde VSCode, abrir `notebooks/03_cnn_propia.ipynb` y ejecutar todas las celdas.  
Cada época toma ~1-2 min en CPU (vs ~4 min con la configuración anterior).

### Paso 3: Si el test no es exitoso (val_acc < 0.35)

Probar ajustes manuales en `src/models/cnn_custom.py`:
- Reducir/aumentar `learning_rate` (ej. 0.0005, 0.0001)
- Cambiar optimizer a `SGD(learning_rate=0.01, momentum=0.9)`
- Reducir dropout a 0.2 o eliminarlo temporalmente
- Aumentar filtros: 64→128→256→512

Luego repetir Paso 1.

---

## IX. PRÓXIMOS PASOS

### AHORA
1. Ejecutar test rápido: `python tests/test_cnn_configs.py --mode fast`
2. Si v7 (o alguna variante) pasa el umbral, ejecutar el notebook completo
3. Continuar con Módulo 4 — Transfer Learning (comparación con ResNet50)

### DESPUÉS
1. Avanzar Módulos 2 a 5 (Data Science) en paralelo con el diseño de la API (Módulo 6) usando un modelo dummy
2. Integrar el modelo final entrenado en la API (Módulo 6)
3. Conectar el Frontend (Módulo 7) a la API real y correr QA end-to-end
4. Cerrar con Módulo 8: respuestas de análisis, documentación y entrega en GitHub

---

**Documento generado**: 10-07-2026
**Versión**: 2.0 (con diagnóstico y mejoras de CNN propia)
**Estado**: Módulo 3 en proceso — ejecutar test para validar
