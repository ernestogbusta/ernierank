from api_key import API_KEY
from flask import Flask, request, jsonify, abort
from robots_parser import EnhancedRobotsParser
from rss_reader import RSSReader
from get_all_links import LinkExtractor
from simple_request import make_request
# Asume que scrape_sitemap está definido en sitemap_crawler.py
from sitemap_crawler import crawl_sitemap

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

    # Inicializa el resultado agrupado
    result = {}

    # Sitemap scraping
    sitemap_data = scrape_sitemap(url)
    result['sitemap'] = sitemap_data if sitemap_data else 'No se pudo raspar la información del sitemap'

    # Robots
    parser = EnhancedRobotsParser(url)
    data = parser.fetch_and_parse()
    result['robots'] = data if data else 'No se pudo raspar la URL proporcionada para robots.txt'

    # RSS
    reader = RSSReader(url)
    if reader.fetch_rss():
        articles = reader.get_articles()
        result['rss'] = {'articles': articles}
    else:
        result['rss'] = 'No se pudo obtener el feed RSS'

    # Links
    extractor = LinkExtractor(url)
    if extractor.fetch_and_extract_links():
        result['links'] = extractor.links
    else:
        result['links'] = 'No se pudieron extraer links de la URL proporcionada'

    # Simple request
    simple_response = make_request(url)
    result['simple_request'] = simple_response if simple_response else 'No se pudo realizar la solicitud simple a la URL proporcionada'

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
