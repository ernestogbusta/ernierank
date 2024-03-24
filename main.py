from flask import Flask, request, jsonify, abort
from sitemap_crawler import crawl_sitemap
from content_extractor import ContentExtractor

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
        extractor = ContentExtractor(site_url)
        content_data = extractor.extract_content()
        if content_data:
            results.append(content_data)
        else:
            results.append({"url": site_url, "error": "Failed to extract content"})

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
