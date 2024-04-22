#!/bin/bash
# Este script inicia el servidor Uvicorn con nuestra aplicación FastAPI y realiza precalentamientos y verificaciones de salud.

# Inicia Uvicorn con nuestra aplicación FastAPI en el puerto especificado.
echo "Iniciando el servidor Uvicorn..."
uvicorn main:app --host=0.0.0.0 --port=${PORT:-10000} &
PID=$!
echo "Servidor iniciado en el PID $PID."

# Función para verificar la salud del servicio.
check_health() {
    echo "Verificando la salud del servicio..."
    if curl -s http://localhost:${PORT:-10000}/health | grep -q 'ok'; then
        echo "Verificación de salud exitosa."
    else
        echo "Fallo en la verificación de salud."
        exit 1
    fi
}

# Espera hasta que el servicio esté operativo verificando su salud.
echo "Esperando que el servicio esté operativo..."
until check_health; do
    sleep 1
done

# Realiza la llamada de pre-calentamiento para preparar el servicio.
echo "Iniciando el precalentamiento del servicio..."
curl -s http://localhost:${PORT:-10000}/preheat > /dev/null
echo "Precalentamiento completado."

# Proceso inicial de carga de URLs para asegurar que el sistema está completamente funcional.
echo "Procesando el primer lote de URLs para precalentar caches y componentes del sistema..."
if curl -s -X POST "http://localhost:${PORT:-10000}/process_urls_in_batches?start_index=0&batch_size=50" | grep -q 'success'; then
    echo "Primer lote de URLs procesado exitosamente."
else
    echo "Fallo al procesar el primer lote de URLs."
    exit 1
fi

echo "Sistema listo para recibir tráfico."
# Mantener el servicio corriendo en el fondo hasta que este termine su ejecución.
wait $PID
