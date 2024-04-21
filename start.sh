#!/bin/bash
# Inicia Uvicorn en segundo plano
uvicorn main:app --host=0.0.0.0 --port=10000 &
PID=$!

# Espera unos segundos para que Uvicorn se inicie
sleep 10

# Realiza la llamada de pre-calentamiento
curl http://localhost:10000/preheat

# Espera que Uvicorn termine (opcional, dependiendo de tu flujo de trabajo)
wait $PID
