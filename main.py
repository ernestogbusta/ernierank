from api_key import API_KEY  # Importa tu clave API desde el módulo api_key
from flask import Flask, request, jsonify, abort
from sitemap_crawler import crawl_sitemap  # Asume que este módulo extrae URLs del sitemap
from content_extractor import ContentExtractor  # Tu módulo fusionado para extracción de contenido

app = Flask(__name__)  # Inicializa tu aplicación Flask

def verify_api_key():
    """Función para verificar si la solicitud incluye una clave API válida."""
    api_key = request.headers.get('X-API-KEY')  # Obtiene la clave API desde el header de la solicitud
    if not api_key or api_key != API_KEY:
        abort(401, description="API Key no válida o no proporcionada.")
        # Si no hay clave API o no coincide, devuelve un error 401

app.before_request(verify_api_key)  # Registra la función para que se ejecute antes de cada solicitud

@app.route('/scrape/all', methods=['POST'])  # Define la ruta y el método permitido
def scrape_all():
    url = request.json.get('url')  # Obtiene la URL del cuerpo de la solicitud JSON
    if not url:
        return jsonify({'error': 'URL no proporcionada'}), 400  # Si no hay URL, devuelve un error 400

    result = {}  # Inicializa el diccionario para los resultados

    # Sitemap scraping
    sitemap_data = crawl_sitemap(url)  # Obtiene los datos del sitemap
    result['sitemap'] = sitemap_data if sitemap_data else 'No se pudo raspar la información del sitemap'

    # Contenido
    content_data = ContentExtractor(url)  # Crea una instancia de ContentExtractor
    extracted_data = content_data.extract_content()  # Extrae el contenido usando el método correspondiente
    if extracted_data:
        result['content'] = extracted_data  # Si se extrajo correctamente, lo añade a los resultados
    else:
        result['content'] = 'No se pudo procesar el contenido de la URL proporcionada'

    return jsonify(result)  # Devuelve los resultados en formato JSON

if __name__ == '__main__':
    app.run(debug=True)  # Ejecuta la aplicación con debug activado para facilitar la depuración
