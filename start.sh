#!/bin/bash
# Este script inicia el servidor Uvicorn con nuestra aplicación FastAPI y realiza precalentamientos y verificaciones de salud.

# Define la función de registro para facilitar el seguimiento de logs.
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

# Inicia Uvicorn con nuestra aplicación FastAPI en el puerto especificado.
log "Iniciando el servidor Uvicorn..."
uvicorn main:app --host=0.0.0.0 --port=${PORT:-10000} &
PID=$!
log "Servidor iniciado en el PID $PID."

# Función para verificar la salud del servicio.
check_health() {
    log "Verificando la salud del servicio..."
    if curl -s --fail http://localhost:${PORT:-10000}/health | grep -q 'ok'; then
        log "Verificación de salud exitosa."
        return 0
    else
        log "Fallo en la verificación de salud."
        return 1
    fi
}

# Espera hasta que el servicio esté operativo verificando su salud.
log "Esperando que el servicio esté operativo..."
until check_health; do
    sleep 1
    log "Reintentando la verificación de salud..."
done

# Realiza la llamada de pre-calentamiento para preparar el servicio.
log "Iniciando el precalentamiento del servicio..."
if curl -s --fail http://localhost:${PORT:-10000}/preheat | grep -q 'ok'; then
    log "Precalentamiento completado."
else
    log "Fallo en el precalentamiento del servicio."
    exit 1
fi

# Proceso inicial de carga de URLs para asegurar que el sistema está completamente funcional.
log "Procesando el primer lote de URLs para precalentar caches y componentes del sistema..."
if curl -s -X POST "http://localhost:${PORT:-10000}/process_urls_in_batches?start=0&batch_size=50" | grep -q 'success'; then
    log "Primer lote de URLs procesado exitosamente."
else
    log "Fallo al procesar el primer lote de URLs."
    exit 1
fi

log "Sistema listo para recibir tráfico."
# Mantener el servicio corriendo en el fondo hasta que este termine su ejecución.
wait $PID
