#!/bin/bash
# Inicia Uvicorn en segundo plano
uvicorn main:app --host=0.0.0.0 --port=${PORT:-10000} &
PID=$!

# Espera unos segundos para que Uvicorn se inicie completamente
sleep 10

# Realiza la llamada de pre-calentamiento hasta que sea exitosa o se intenten 5 veces
attempts=0
max_attempts=5
while [ $attempts -lt $max_attempts ]; do
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT:-10000}/preheat)
    if [ "$response" -eq 200 ]; then
        echo "Preheat successful"
        break
    else
        echo "Preheat failed with status $response, retrying..."
        ((attempts++))
        sleep 5
    fi
done

if [ $attempts -eq $max_attempts ]; then
    echo "Failed to preheat after $max_attempts attempts."
    exit 1
fi

# Espera que Uvicorn termine (opcional, dependiendo de tu flujo de trabajo)
wait $PID
