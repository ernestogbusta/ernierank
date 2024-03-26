from flask import Flask, request, jsonify
from sitemap_crawler import SitemapExtractor
from content_extractor import SEOContentAnalyzer

app = Flask(__name__)

@app.route('/scrape', methods=['POST'])
def scrape_site():
    # Obtén la URL del cuerpo de la solicitud
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "URL not provided"}), 400
    
    # Inicia el extractor de sitemap con la URL proporcionada
    sitemap_extractor = SitemapExtractor(url)
    urls_found = sitemap_extractor.crawl_sitemap()
    
    if not urls_found:
        return jsonify({"error": "No URLs found in the sitemap"}), 404
    
    # Analiza el contenido de cada URL encontrada
    results = []
    for site_url in urls_found:
        analyzer = SEOContentAnalyzer(site_url)
        content_analysis = analyzer.analyze_content()
        if content_analysis:
            results.append({"url": site_url, **content_analysis})
        else:
            results.append({"url": site_url, "error": "Failed to analyze content"})
    
    return jsonify(results)

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test endpoint is working!"})

if __name__ == '__main__':
    app.run(debug=True)
