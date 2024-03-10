from flask import Flask, request, jsonify
# Asegúrate de que las importaciones reflejen la estructura actual de tus módulos y clases
from robots_parser import EnhancedRobotsParser
from rss_reader import RSSReader
from get_all_links import LinkExtractor  # Actualizado para usar la clase LinkExtractor
from simple_request import make_request

app = Flask(__name__)

@app.route('/scrape/robots', methods=['POST'])
def scrape_robots():
    url = request.json.get('url')
    if not url:
        return jsonify({'error': 'URL no proporcionada'}), 400
    parser = EnhancedRobotsParser(url)
    data = parser.fetch_and_parse()
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'No se pudo raspar la URL proporcionada'}), 500

@app.route('/scrape/rss', methods=['POST'])
def scrape_rss():
    url = request.json.get('url')
    if not url:
        return jsonify({'error': 'URL no proporcionada'}), 400
    reader = RSSReader(url)
    if reader.fetch_rss():
        articles = reader.get_articles()
        return jsonify({'articles': articles})
    else:
        return jsonify({'error': 'No se pudo obtener el feed RSS'}), 500

@app.route('/scrape/links', methods=['POST'])
def scrape_links():
    url = request.json.get('url')
    if not url:
        return jsonify({'error': 'URL no proporcionada'}), 400
    extractor = LinkExtractor(url)
    if extractor.fetch_and_extract_links():
        return jsonify({'links': extractor.links})
    else:
        return jsonify({'error': 'No se pudieron extraer links de la URL proporcionada'}), 500

@app.route('/scrape/simple_request', methods=['POST'])
def simple_request_route():
    url = request.json.get('url')
    if not url:
        return jsonify({'error': 'URL no proporcionada'}), 400
    response = make_request(url)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
