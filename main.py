from flask import Flask, request, jsonify, abort
from sitemap_crawler import crawl_sitemap
from content_extractor import SEOContentAnalyzer  # Corrige el nombre de la importación

app = Flask(__name__)

@app.route('/scrape', methods=['POST'])
def scrape_site():
    data = request.json
    url = data.get('url')
    if not url:
        abort(400, description="No URL provided")
    
    try:
        sitemap_urls = crawl_sitemap(url)
        if not sitemap_urls:
            abort(500, description="Could not retrieve any URLs from the sitemap")
        
        results = []
        for site_url in sitemap_urls:
            analyzer = SEOContentAnalyzer(site_url)
            analysis_results = analyzer.analyze_content()
            if analysis_results:
                results.append({**analysis_results, 'url': site_url})
            else:
                results.append({"url": site_url, "error": "Failed to analyze content"})
        
        return jsonify(results)
    except Exception as e:
        abort(500, description=str(e))

if __name__ == "__main__":
    app.run(debug=True)
