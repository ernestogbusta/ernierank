#!/bin/bash
# Inicia Uvicorn con configuración de host y puerto
PORT=${PORT:-10000}
echo "Starting Uvicorn on port $PORT"
uvicorn main:app --host=0.0.0.0 --port=$PORT &
PID=$!

# Espera unos segundos para que Uvicorn se inicie
echo "Waiting for Uvicorn to start"
sleep 10

# Realiza la llamada de pre-calentamiento
echo "Sending preheat request"
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/preheat)
if [ "$response" -eq 200 ]; then
    echo "Preheat successful"
else
    echo "Preheat failed with status $response"
fi

# Espera que Uvicorn termine (opcional, dependiendo de tu flujo de trabajo)
wait $PID
