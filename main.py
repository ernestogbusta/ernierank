from flask import Flask, request, jsonify
from robots_parser import EnhancedRobotsParser
from rss_reader import RSSReader
from get_all_links import LinkExtractor
from simple_request import make_request

app = Flask(__name__)

@app.route('/scrape/all', methods=['POST'])
def scrape_all():
    url = request.json.get('url')
    if not url:
        return jsonify({'error': 'URL no proporcionada'}), 400

    # Inicializa el resultado agrupado
    result = {}

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
