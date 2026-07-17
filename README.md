# BrainTumor Detector

Sistema de deteccion de tumores cerebrales en imagenes MRI usando redes neuronales convolucionales (CNN).

---

## Arquitectura del Sistema

`
                            +-------------------+
                            |   Frontend HTML   |
                            | (profesional.html)|
                            +--------+----------+
                                     | HTTP POST
                                     | image (multipart)
                                     v
+------------------+       +-------------------+       +------------------+
|   Notebooks ML   | ----> |   Flask API       | ----> |   Modelo CNN     |
| (EDA, CNN, etc)  |       | POST /predict     |       | cnn_custom.keras |
+------------------+       +-------------------+       +------------------+
                                     |
                          +----------+-----------+
                          |                      |
                   +-----v------+        +-------v------+
                   |   Docker   |        |   systemd    |
                   |  contenedor|        |  servicio    |
                   +------------+        +--------------+
`

## Resultados del Modelo

| Metrica | CNN Propia | ResNet50 |
|---------|-----------|----------|
| Accuracy | **79.25%** | 52.50% |
| Precision | **82.21%** | 57.36% |
| Recall | **79.25%** | 52.50% |
| F1-Score | **78.80%** | 49.46% |

**Modelo en produccion:** models/cnn_custom.keras

## API (Rol 3)

### Endpoint

`ash
POST /predict
Content-Type: multipart/form-data
Body: image=<archivo MRI>
`

### Respuesta

`json
{
  "tipo_tumor": "glioma",
  "probabilidad": 0.987,
  "probabilidades": {
    "glioma": 0.0004,
    "meningioma": 0.0126,
    "notumor": 0.9870,
    "pituitary": 0.0000
  }
}
`

### Ejecutar localmente

`ash
cd api
pip install -r requirements.txt
python app.py
`

## Despliegue

### Docker

`ash
docker build -t braintumor-api .
docker run -d --name braintumor-api -p 5000:5000 braintumor-api
docker logs braintumor-api
`

### systemd (Linux)

`ash
sudo cp braintumor-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable braintumor-api
sudo systemctl start braintumor-api
sudo journalctl -u braintumor-api -f
`

## Frontend

Dos versiones disponibles:

1. **Original:** rontend/index.html
2. **Profesional:** rontend/profesional.html (recomendada - incluye imagenes decorativas, radar chart y animaciones)

Abrir el archivo directamente en el navegador. La API debe estar corriendo en localhost:5000.

## Notebooks

| Notebook | Contenido |
|----------|-----------|
| 01_eda.ipynb | Analisis exploratorio de datos |
| 02_preprocesamiento.ipynb | Preprocesamiento y augmentation |
| 03_cnn_propia.ipynb | Entrenamiento de CNN desde cero |
| 04_transfer_learning.ipynb | Transfer Learning con ResNet50 |
| 05_evaluacion.ipynb | Evaluacion comparativa |
| 06_visualizaciones_mejoradas.ipynb | Visualizaciones profesionales |

## Estructura del Proyecto

`
.
+-- api/                  # API Flask
|   +-- app.py            # Endpoint POST /predict
|   +-- inference.py      # Carga de modelo y prediccion
|   +-- requirements.txt  # Dependencias Python
+-- frontend/             # Interfaz web
|   +-- index.html        # Version original
|   +-- profesional.html  # Version mejorada
|   +-- profesional.css   # Estilos
|   +-- profesional.js    # Logica (incluye radar chart)
|   +-- assets/           # Imagenes de ejemplo
+-- notebooks/            # Notebooks Jupyter
+-- models/               # Modelos entrenados
+-- src/                  # Codigo fuente ML
+-- docs/                 # Documentacion y graficos
|   +-- despliegue.md     # Guia de despliegue
|   +-- informe.html      # Informe del proyecto
+-- Dockerfile            # Despliegue con Docker
+-- braintumor-api.service # Servicio systemd
+-- README.md             # Este archivo
`

## Equipo

| Rol | Integrante | Modulos |
|-----|-----------|---------|
| Rol 1 | Fabricio Vasquez | EDA, Preprocesamiento, CNN propia |
| Rol 2 | Jhordyn Velasquez | Transfer Learning |
| Rol 3 | Eduardo Zuta | Evaluacion, API, Despliegue |
| Rol 4 | Jhordy Torrejon | Frontend, Documentacion |

## Disclaimer

**Aviso importante:** Este sistema es una herramienta de apoyo y segunda opinion. No reemplaza el diagnostico de un profesional medico. Consulte siempre a un especialista.
