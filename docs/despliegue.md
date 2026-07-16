# Despliegue - Brain Tumor API

## Opción A: Docker

```bash
# Construir imagen
docker build -t braintumor-api .

# Ejecutar contenedor
docker run -d --name braintumor-api -p 5000:5000 braintumor-api

# Verificar logs
docker logs braintumor-api

# Probar endpoint
curl -X POST -F "image=@ruta/a/imagen.jpg" http://localhost:5000/predict

# Detener y eliminar
docker stop braintumor-api && docker rm braintumor-api
```

## Opción B: systemd (Linux sin Docker)

```bash
# Crear directorio y virtualenv
sudo mkdir -p /opt/braintumor-api
sudo cp -r api/ models/cnn_custom.keras /opt/braintumor-api/
cd /opt/braintumor-api
python3 -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt

# Copiar archivo de servicio
sudo cp braintumor-api.service /etc/systemd/system/

# Recargar systemd e iniciar servicio
sudo systemctl daemon-reload
sudo systemctl enable braintumor-api
sudo systemctl start braintumor-api

# Verificar estado
sudo systemctl status braintumor-api

# Ver logs en tiempo real
sudo journalctl -u braintumor-api -f

# Ver últimas 50 líneas de log
sudo journalctl -u braintumor-api -n 50 --no-pager
```

## Prueba de la API

```bash
curl -X POST \
  -F "image=@ruta/al/glioma/imagen.jpg" \
  http://localhost:5000/predict

# Respuesta esperada:
# {"tipo_tumor":"glioma","probabilidad":0.95,"probabilidades":{...}}
```
