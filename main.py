from flask import Flask, request, jsonify, abort
import nltk

# Descarga de los recursos de NLTK necesarios para TextBlob
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown') 

from sitemap_crawler import crawl_sitemap
from content_extractor import SEOContentAnalyzer  # Asume que esta es la clase actualizada

app = Flask(__name__)

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
        
        # Aquí se implementaría la lógica específica del análisis de canibalización SEO.
        # Por ejemplo, este código podría adaptarse para comparar keywords entre páginas.
        # Por ahora, simplemente devolveremos las URLs extraídas.
        
        return jsonify({"urls": sitemap_urls})
    except Exception as e:
        abort(500, description=f"Se produjo un error: {e}")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
