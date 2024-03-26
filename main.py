from flask import Flask, request, jsonify
from sitemap_crawler import SitemapExtractor
from content_extractor import SEOContentAnalyzer

app = Flask(__name__)

@app.route('/scrape', methods=['POST'])
def scrape_site():
    data = request.json
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    domain_url = data['url']
    sitemap_extractor = SitemapExtractor(domain_url)
    urls = sitemap_extractor.crawl_sitemap()

    if not urls:
        return jsonify({"error": "No URLs found in the sitemap"}), 404

    results = []
    for url in urls:
        analyzer = SEOContentAnalyzer(url)
        content_data = analyzer.analyze_content()
        if content_data:
            results.append({"url": url, "data": content_data})
        else:
            results.append({"url": url, "error": "Failed to analyze content"})

    return jsonify(results), 200

if __name__ == '__main__':
    app.run(debug=True)
