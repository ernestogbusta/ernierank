from flask import Flask, request, jsonify, abort
from content_extractor import ContentExtractor
from sitemap_crawler import crawl_sitemap

app = Flask(__name__)

API_KEY = "fba647b41ae2483bd9d4dc19bd90ab94"  # Asegúrate de reemplazar esto con tu propia clave API

def verify_api_key():
    """Verifica que la solicitud contenga una clave API válida."""
    api_key = request.headers.get('X-API-KEY')
    if not api_key or api_key != API_KEY:
        abort(401, description="API Key no válida o no proporcionada.")

app.before_request(verify_api_key)

@app.route('/scrape/all', methods=['POST'])
def scrape_all():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL no proporcionada'}), 400

    url = data['url']
    result = {}

    # Extracción de contenido
    extractor = ContentExtractor(url)
    content_data = extractor.extract_content()
    if content_data:
        result['content'] = content_data
    else:
        result['content_error'] = "No se pudo extraer el contenido."

    # Rastreo del sitemap
    sitemap_urls = crawl_sitemap(url)
    if sitemap_urls:
        result['sitemap_urls'] = sitemap_urls
    else:
        result['sitemap_error'] = "No se pudo rastrear el sitemap o no se encontró."

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
