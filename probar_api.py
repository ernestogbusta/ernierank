import requests

# URL de la API en Render
url = "https://ernierank-vd20.onrender.com/process_urls_in_batches"

# Cabeceras (headers)
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

# Datos a enviar
payload = {
    "domain": "https://www.clinicatambre.com/",
    "batch_size": 100,
    "start": 0
}

# Hacemos la petición POST
response = requests.post(url, headers=headers, json=payload)

# Mostramos el status code y la respuesta
print(f"Status Code: {response.status_code}")
print(response.text)
