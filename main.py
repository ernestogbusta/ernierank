from flask import Flask, request, jsonify, abort
import nltk
from content_extractor import SEOContentAnalyzer
from sitemap_crawler import crawl_sitemap

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')

app = Flask(__name__)

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    data = request.json
    url = data.get('url')
    if not url:
        abort(400, description="No se proporcionó ninguna URL para analizar")

    try:
        analyzer = SEOContentAnalyzer(url)
        analysis_results = analyzer.analyze_content()
        if analysis_results:
            return jsonify(analysis_results)
        else:
            abort(500, description="No se pudo analizar el contenido de la URL")
    except Exception as e:
        abort(500, description=f"Ocurrió un error: {e}")

@app.route('/analyze_canibalizacion', methods=['POST'])
def analyze_canibalizacion():
    data = request.json
    url = data.get('url')
    if not url:
        abort(400, description="No se proporcionó ninguna URL para analizar la canibalización SEO")

    try:
        # Aquí puedes implementar la lógica para analizar la canibalización SEO
        # utilizando las clases y funciones necesarias.
        
        # Devuelve los resultados apropiados
        
    except Exception as e:
        abort(500, description=f"Ocurrió un error: {e}")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
