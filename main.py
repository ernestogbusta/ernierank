# main.py

from flask import Flask, request, jsonify, abort
from sitemap_crawler import SitemapExtractor
from content_extractor import SEOContentAnalyzer

app = Flask(__name__)

@app.route('/scrape', methods=['POST'])
def scrape_site():
    data = request.json
    url = data.get('url')
    if not url:
        abort(400, description="URL not provided")
    
    sitemap_extractor = SitemapExtractor(url)
    sitemap_urls = sitemap_extractor.crawl_sitemap()
    if not sitemap_urls:
        abort(500, description="Could not retrieve URLs from sitemap")
    
    results = []
    for site_url in sitemap_urls:
        analyzer = SEOContentAnalyzer(site_url)
        content_analysis = analyzer.analyze_content()
        if content_analysis:
            results.append({'url': site_url, **content_analysis})
        else:
            results.append({'url': site_url, 'error': 'Could not analyze content'})
    
    return jsonify(results)

if __name__ == "__main__":
    app.run()
