from flask import Flask, request, jsonify
from sitemap_crawler import SitemapExtractor
from content_extractor import SEOContentAnalyzer

app = Flask(__name__)

@app.route('/scrape', methods=['POST'])
def scrape_site():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL is required.'}), 400

    try:
        sitemap_extractor = SitemapExtractor(data['url'])
        urls = sitemap_extractor.crawl_sitemap()
        results = []
        for url in urls:
            analyzer = SEOContentAnalyzer(url)
            analysis = analyzer.analyze_content()
            results.append(analysis)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
