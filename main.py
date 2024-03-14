from api_key import API_KEY
from flask import Flask, request, jsonify, abort
from sitemap_crawler import crawl_sitemap
# Asume que ContentExtractor es tu nuevo módulo fusionado que maneja la extracción de contenido
from content_extractor import ContentExtractor

app = Flask(__name__)

def verify_api_key():
    """Verifica que la solicitud contenga una API key válida."""
    api_key = request.headers.get('X-API-KEY')
    if not api_key or api_key != API_KEY:
        abort(401, description="API Key no válida o no proporcionada.")

app.before_request(verify_api_key)

@app.route('/scrape/all', methods=['POST'])
def scrape_all():
    url = request.json.get('url')
    if not url:
        return jsonify({'error': 'URL no proporcionada'}), 400

    result = {}

    # Sitemap scraping
    sitemap_data = crawl_sitemap(url)
    result['sitemap'] = sitemap_data if sitemap_data else 'No se pudo raspar la información del sitemap'

    # Contenido
    content_data = ContentExtractor(url)  # Suponiendo que ContentExtractor maneja todo
    extracted_data = content_data.extract_content()  # Asegúrate de ajustar este método a tu implementación real
    if extracted_data:
        result['content'] = extracted_data
    else:
        result['content'] = 'No se pudo procesar el contenido de la URL proporcionada'

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
