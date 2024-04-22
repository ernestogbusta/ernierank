#!/bin/bash
echo "Iniciando el servidor Uvicorn..."
uvicorn main:app --host=0.0.0.0 --port=${PORT:-10000} &
PID=$!
echo "Servidor iniciado en el PID $PID."

check_health() {
    echo "Verificando la salud del servicio..."
    response=$(curl -s http://localhost:${PORT:-8000}/health)
    echo "Respuesta de salud: $response"
    if echo "$response" | grep -q '"status":"ok"'; then
        echo "Verificación de salud exitosa."
        return 0
    else
        echo "Fallo en la verificación de salud."
        return 1
    fi
}

echo "Esperando que el servicio esté operativo..."
until check_health; do
    sleep 1
    echo "Reintentando la verificación de salud..."
done

echo "Iniciando el precalentamiento del servicio..."
if curl -s http://localhost:${PORT:-8000}/preheat | grep -q '"status":"ok"'; then
    echo "Precalentamiento completado exitosamente."
else
    echo "Fallo en el precalentamiento del servicio."
    exit 1
fi

echo "Sistema listo para recibir tráfico."
wait $PID
