#!/bin/bash
# Este script inicia el servidor Uvicorn con nuestra aplicación FastAPI y realiza precalentamientos y verificaciones de salud.

echo "Iniciando el servidor Uvicorn..."
uvicorn main:app --host=0.0.0.0 --port=${PORT:-10000} &
PID=$!
echo "Servidor iniciado en el PID $PID."

# Función para verificar la salud del servicio.
check_health() {
    echo "Verificando la salud del servicio..."
    response=$(curl -s http://localhost:${PORT:-10000}/health)
    echo "Respuesta de salud: $response"
    if echo "$response" | grep -q 'ok'; then
        echo "Verificación de salud exitosa."
        return 0
    else
        echo "Fallo en la verificación de salud."
        return 1
    fi
}

# Espera hasta que el servicio esté operativo verificando su salud.
echo "Esperando que el servicio esté operativo..."
until check_health; do
    sleep 1
    echo "Reintentando la verificación de salud..."
done

echo "Iniciando el precalentamiento del servicio..."
if curl -s http://localhost:${PORT:-10000}/preheat | grep -q 'ok'; then
    echo "Precalentamiento completado exitosamente."
else
    echo "Fallo en el precalentamiento del servicio."
    exit 1
fi

echo "Procesando el primer lote de URLs para precalentar caches y componentes del sistema..."
# Asegúrate de que la URL y parámetros aquí coincidan con cómo tu API está esperando recibir las llamadas
if curl -s -X POST "http://localhost:${PORT:-10000}/process_urls_in_batches" -H "Content-Type: application/json" -d '{"start":0,"batch_size":50}' | grep -q 'success'; then
    echo "Primer lote de URLs procesado exitosamente."
else
    echo "Fallo al procesar el primer lote de URLs."
    exit 1
fi

echo "Sistema listo para recibir tráfico."
wait $PID
