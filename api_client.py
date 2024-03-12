import requests

def hacer_solicitud_a_api(url_destino):
    api_url = "https://ernierank-vd20.onrender.com/scrape/all"  # Asegúrate de reemplazar esto con la URL real de tu API
    API_KEY = "tu_api_key_secreta"  # Asegúrate de reemplazar esto con tu API key real
    headers = {"Authorization": f"Bearer fba647b41ae2483bd9d4dc19bd90ab94"}
    data = {"url": url_destino}

    response = requests.post(api_url, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error al llamar a la API: {response.status_code}, Respuesta: {response.text}")
