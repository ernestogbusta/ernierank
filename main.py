from flask import Flask, request, jsonify, abort
import os
import logging
import nltk
from sitemap_crawler import crawl_sitemap
from content_extractor import SEOContentAnalyzer

# Configuración inicial
app = Flask(__name__)

# Configurar el nivel de logging desde una variable de entorno
log_level = os.environ.get('LOG_LEVEL', 'WARNING').upper()
app.logger.setLevel(log_level)

# Descarga de los recursos de NLTK necesarios para TextBlob (considerar mover fuera de main.py en producción)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')

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
        abort(400, description="No se proporcionó URL")
    
    try:
        sitemap_urls = crawl_sitemap(url)
        if not sitemap_urls:
            abort(500, description="No se pudieron recuperar URLs del sitemap")
        
        results = []
        for site_url in sitemap_urls:
            analyzer = SEOContentAnalyzer(site_url)
            analysis_results = analyzer.analyze_content()
            if analysis_results:
                results.append({**analysis_results, 'url': site_url})
            else:
                results.append({"url": site_url, "error": "No se pudo analizar el contenido"})
        
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Se produjo un error inesperado: {e}")
        abort(500, description="Error interno del servidor")

@app.route('/analisis-canibalizacion', methods=['POST'])
def analisis_canibalizacion():
    data = request.json
    urls = data.get('urls')
    if not urls:
        abort(400, description="URLs no proporcionadas")
    
    try:
        # Aquí implementarías tu lógica de análisis de canibalización
        
        return jsonify({"urls": urls})  # Modificar según la implementación real
    except Exception as e:
        app.logger.error(f"Se produjo un error en el análisis de canibalización: {e}")
        abort(500, description="Error interno del servidor")

if __name__ == "__main__":
    app.run()
