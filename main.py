from flask import Flask, request, jsonify, abort
import os
import logging
import nltk
# Importa la clase SitemapExtractor en lugar de una función
from sitemap_crawler import SitemapExtractor
from content_extractor import SEOContentAnalyzer

app = Flask(__name__)
log_level = os.environ.get('LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(level=log_level)

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('brown', quiet=True)

@app.errorhandler(400)
def bad_request(error):
    return jsonify(error=str(error)), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify(error=str(error)), 500

@app.route('/scrape', methods=['POST'])
def scrape_site():
    data = request.json
    url = data.get('url')
    if not url:
        abort(400, description="URL no proporcionada")
    
    # Crea una instancia de SitemapExtractor con la URL del dominio
    sitemap_extractor = SitemapExtractor(url)
    # Llama al método crawl_sitemap sobre la instancia
    sitemap_urls = sitemap_extractor.crawl_sitemap()
    if not sitemap_urls:
        abort(500, description="No se pudieron recuperar URLs del sitemap")
    
    results = []
    for site_url in sitemap_urls:
        analyzer = SEOContentAnalyzer(site_url)
        content_analysis = analyzer.analyze_content()
        if content_analysis:
            result = {'url': site_url, **content_analysis}
        else:
            result = {'url': site_url, 'error': 'No se pudo analizar el contenido'}
        results.append(result)
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
