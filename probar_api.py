import requests
import json
import time
import random

# URL de la API desplegada en Render
api_url = "https://ernierank-vd20.onrender.com/process_urls_in_batches"

# Configura el dominio que quieres analizar
domain = "https://aulacm.com/"

# Cabeceras HTTP necesarias
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
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

    # 💤 Pausa aleatoria entre 1.5 y 4 segundos
    sleep_time = random.uniform(1.5, 4.0)
    print(f"⏳ Pausando {sleep_time:.2f} segundos para respetar el servidor...")
    time.sleep(sleep_time)

# Guardar todos los resultados en un archivo JSON
output_filename = "resultados_rastreo.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

print(f"\n🎯 Resultado completo guardado en: {output_filename}")