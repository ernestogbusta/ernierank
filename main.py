from flask import Flask, request, jsonify, abort
from sitemap_crawler import crawl_sitemap
from content_extractor import SEOContentAnalyzer  # Asegúrate de que este importe sea correcto

app = Flask(__name__)

@app.route('/scrape', methods=['POST'])
def scrape_site():
    data = request.json
    url = data.get('url')
    
    if not url:
        abort(400, description="No URL provided")
    
    try:
        sitemap_urls = crawl_sitemap(url)
    except Exception as e:
        abort(500, description=str(e))
    
    if not sitemap_urls:
        abort(500, description="Could not retrieve any URLs from the sitemap")
    
    results = []
    for site_url in sitemap_urls:
        analyzer = SEOContentAnalyzer(site_url)  # Actualiza esta línea para usar la nueva clase
        analysis_results = analyzer.analyze_content()  # Asegúrate de que este método exista y haga lo que esperas
        if analysis_results:
            results.append(analysis_results)
        else:
            results.append({"url": site_url, "error": "Failed to extract content"})

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
