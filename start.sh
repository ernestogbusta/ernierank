#!/bin/bash
# Inicia Uvicorn en segundo plano
uvicorn main:app --host=0.0.0.0 --port=${PORT:-10000} &
PID=$!

# Espera hasta que Uvicorn esté completamente operativo
echo "Esperando que Uvicorn esté completamente operativo..."
while ! curl -s http://localhost:${PORT:-10000}/health; do   
    sleep 1
    echo "Reintentando..."
done

echo "Uvicorn está operativo. Iniciando precalentamiento..."
# Realiza la llamada de pre-calentamiento
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT:-10000}/preheat)
if [ "$response" -eq 200 ]; then
    echo "Precalentamiento exitoso. Realizando segunda verificación..."
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT:-10000}/preheat)
    if [ "$response" -eq 200 ]; then
        echo "Segunda verificación exitosa. Sistema listo."
    else
        echo "La segunda verificación falló. Estado: $response"
        exit 1
    fi
else
    echo "Precalentamiento fallido con estado $response"
    exit 1
fi

# Mantener el servicio corriendo
wait $PID
