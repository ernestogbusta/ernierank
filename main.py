from flask import Flask, request, jsonify, abort
from sitemap_crawler import crawl_sitemap
from content_extractor import ContentExtractor

# Suponiendo que API_KEY está definida en api_key.py o directamente aquí
API_KEY = "fba647b41ae2483bd9d4dc19bd90ab94"

app = Flask(__name__)

def verify_api_key():
    """Verifica que la solicitud contenga una clave API válida."""
    api_key = request.headers.get('X-API-KEY')
    if not api_key or api_key != API_KEY:
        abort(401, description="API Key no válida o no proporcionada.")

app.before_request(verify_api_key)

@app.route('/scrape/all', methods=['POST'])
def scrape_all():
    data = request.json
    if 'url' not in data:
        return jsonify({'error': 'URL not provided'}), 400

    url = data['url']
    result = {}

    # Sitemap scraping
    sitemap_urls = crawl_sitemap(url)
    if sitemap_urls:
        result['sitemap_urls'] = sitemap_urls
    else:
        result['sitemap_error'] = "Failed to retrieve or parse sitemap."

    # Contenido
    content_extractor = ContentExtractor(url)
    content_data = content_extractor.extract_content()
    if content_data:
        result.update(content_data)  # Se espera que content_data sea un diccionario con 'links', 'headings', y 'body_text'
    else:
        result['content_error'] = "Failed to process the URL content."

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
