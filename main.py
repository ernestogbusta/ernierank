from flask import Flask, request, jsonify, abort
import os
import logging
import nltk
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
    
    try:
        sitemap_extractor = SitemapExtractor(url)
        sitemap_urls = sitemap_extractor.crawl_sitemap()
        if not sitemap_urls:
            abort(500, description="No se pudieron recuperar URLs del sitemap")
        
        results = []
        for site_url in sitemap_urls:
            try:
                analyzer = SEOContentAnalyzer(site_url)
                content_analysis = analyzer.analyze_content()
                if content_analysis:
                    result = {'url': site_url, **content_analysis}
                else:
                    result = {'url': site_url, 'error': 'No se pudo analizar el contenido'}
                results.append(result)
            except Exception as e:
                app.logger.error(f"Error al analizar {site_url}: {e}")
                results.append({'url': site_url, 'error': 'Error durante el análisis de contenido'})
        
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Error general al procesar la solicitud: {e}")
        abort(500, description="Error interno del servidor")

if __name__ == "__main__":
    app.run(debug=True)
