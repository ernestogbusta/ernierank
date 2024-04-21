#!/bin/bash
# Inicia Uvicorn en segundo plano
uvicorn main:app --host=0.0.0.0 --port=${PORT:-10000} &
PID=$!

# Espera unos segundos para que Uvicorn se inicie
sleep 10

# Realiza la llamada de pre-calentamiento
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:10000/preheat)
if [ "$response" -eq 200 ]; then
    echo "Preheat successful"
else
    echo "Preheat failed with status $response"
fi

# Espera que Uvicorn termine (opcional, dependiendo de tu flujo de trabajo)
wait $PID
