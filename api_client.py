import requests

def hacer_solicitud_a_api(url_destino):
    api_url = "https://ernierank-vd20.onrender.com/scrape/all"
    API_KEY = "fba647b41ae2483bd9d4dc19bd90ab94"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {"url": url_destino}

    try:
        response = requests.post(api_url, json=data, headers=headers)
        response.raise_for_status()  # Esto asegura que se manejen respuestas con error.
        return response.json()
    except requests.HTTPError as e:
        # Manejo específico de errores HTTP, p. ej., 404 Not Found, 500 Internal Server Error.
        print(f"Error HTTP al llamar a la API: {e}")
    except requests.RequestException as e:
        # Manejo de otros errores de solicitud, p. ej., problemas de red.
        print(f"Error al llamar a la API: {e}")
    return None
