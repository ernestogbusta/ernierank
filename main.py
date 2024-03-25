from flask import Flask, request, jsonify, abort
import os
import logging
import nltk
from sitemap_crawler import crawl_sitemap  # Asume que esta función devuelve una lista de URLs
from content_extractor import SEOContentAnalyzer  # Asume que esta clase tiene un método analyze_content() que devuelve un diccionario

# Configuración inicial
app = Flask(__name__)

# Configurar el nivel de logging desde una variable de entorno
log_level = os.environ.get('LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(level=log_level)

# Descarga de los recursos de NLTK necesarios para TextBlob
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
        sitemap_urls = crawl_sitemap(url)  # Obtiene las URLs desde el sitemap del dominio dado
        if not sitemap_urls:
            abort(500, description="No se pudieron recuperar URLs del sitemap")
        
        # Analiza el contenido de cada URL encontrada en el sitemap
        results = []
        for site_url in sitemap_urls:
            analyzer = SEOContentAnalyzer(site_url)
            analysis_results = analyzer.analyze_content()
            if analysis_results:
                results.append({'url': site_url, **analysis_results})
            else:
                results.append({"url": site_url, "error": "No se pudo analizar el contenido"})
        
        return jsonify(results)
    except Exception as e:
        logging.error(f"Se produjo un error inesperado: {e}")
        abort(500, description="Error interno del servidor")

if __name__ == "__main__":
    app.run(debug=True)  # Cambia debug a False en producción
