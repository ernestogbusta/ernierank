from api_key import API_KEY
from flask import Flask, request, jsonify, abort
from sitemap_crawler import crawl_sitemap
from content_extractor import ContentExtractor

app = Flask(__name__)

def verify_api_key():
    """Verifica que la solicitud contenga una clave API válida."""
    api_key = request.headers.get('X-API-KEY')
    if not api_key or api_key != API_KEY:
        abort(401, description="API Key no válida o no proporcionada.")

app.before_request(verify_api_key)

@app.route('/scrape/all', methods=['POST'])
def scrape_all():
    body = request.get_json()
    if not body or 'url' not in body:
        return jsonify({'error': 'URL no proporcionada'}), 400

    url = body['url']
    result = {}

    # Sitemap scraping
    sitemap_urls, sitemap_error = crawl_sitemap(url)
    if sitemap_urls:
        result['sitemap_urls'] = sitemap_urls
    else:
        result['sitemap_error'] = sitemap_error or "Error al obtener URLs del sitemap."

    # Contenido
    content_extractor = ContentExtractor(url)
    content_data, content_error = content_extractor.extract_content()
    if content_data:
        result.update(content_data)  # Se espera que content_data sea un diccionario
    else:
        result['content_error'] = content_error or "Error al procesar el contenido de la URL."

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
