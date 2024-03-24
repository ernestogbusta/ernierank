from flask import Flask, request, jsonify, abort
import nltk

# Descarga de los recursos de NLTK necesarios para TextBlob
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')

from sitemap_crawler import crawl_sitemap
from content_extractor import SEOContentAnalyzer  # Asume que esta es la clase actualizada

app = Flask(__name__)

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
        abort(500, description=f"Se produjo un error: {e}")

@app.route('/analisis-canibalizacion', methods=['POST'])
def analisis_canibalizacion():
    data = request.json
    url = data.get('url')
    if not url:
        abort(400, description="URL no proporcionada")
    
    try:
        sitemap_urls = crawl_sitemap(url)
        if not sitemap_urls:
            abort(500, description="No se pudieron recuperar URLs del sitemap")
        
        # Implementa aquí la lógica específica del análisis de canibalización SEO
        
        return jsonify({"urls": sitemap_urls})
    except Exception as e:
        abort(500, description=f"Se produjo un error: {e}")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
