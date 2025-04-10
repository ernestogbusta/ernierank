import requests
import json
import time

# URL de la API desplegada en Render
api_url = "https://ernierank-vd20.onrender.com/process_urls_in_batches"

# Configura el dominio que quieres analizar
domain = "https://aulacm.com/"

# Cabeceras HTTP necesarias
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

# Configuración inicial
batch_size = 10
start = 0

# Para almacenar todos los resultados
all_results = []

while True:
    payload = {
        "domain": domain,
        "batch_size": batch_size,
        "start": start
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()  # Lanza error si no es 2xx
    except requests.RequestException as e:
        print(f"❌ Error haciendo la solicitud: {e}")
        break

    data = response.json()

    # Añadir resultados de este batch
    batch_results = data.get("processed_urls", [])
    all_results.extend(batch_results)

    print(f"✅ Batch procesado: start={start}, URLs procesadas en este batch: {len(batch_results)}")

    # Verificar si hay más batches
    if not data.get("more_batches", False):
        print("🏁 No hay más batches. Terminando...")
        break

    # Preparar el siguiente batch
    start = data.get("next_batch_start", 0)

    # Pequeña pausa entre batches para ser más respetuoso
    time.sleep(2)

# Guardar todos los resultados en un archivo JSON
output_filename = "resultados_rastreo.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

print(f"\n🎯 Resultado completo guardado en: {output_filename}")