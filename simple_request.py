import requests
import sys

def make_request(url):
    """
    Hace una solicitud HTTP GET a la URL proporcionada y imprime el contenido,
    la URL final y el código de estado de la respuesta.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        print("Contenido de la respuesta:\n", response.text[:500])  # Imprime los primeros 500 caracteres
        print("\nURL final:", response.url)
        print("\nCódigo de estado:", response.status_code)
    except requests.exceptions.RequestException as e:
        print(f"La solicitud HTTP GET ha fallado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python simple_request.py <URL>")
        sys.exit(1)
    else:
        url_provided = sys.argv[1]
        make_request(url_provided)
