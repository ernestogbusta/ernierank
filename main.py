from flask import Flask, request, jsonify, abort
from sitemap_crawler import crawl_sitemap
from content_extractor import ContentExtractor

app = Flask(__name__)

@app.route('/scrape', methods=['POST'])
def scrape_site():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    sitemap_urls = crawl_sitemap(url)
    
    if not sitemap_urls:
        return jsonify({"error": "Could not retrieve any URLs from the sitemap"}), 500
    
    results = []
    for site_url in sitemap_urls:
        extractor = ContentExtractor(site_url)
        content_data = extractor.extract_content()
        if content_data:
            results.append(content_data)
        else:
            results.append({"url": site_url, "error": "Failed to extract content"})

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
