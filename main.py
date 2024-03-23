from flask import Flask, request, jsonify
from sitemap_crawler import crawl_sitemap

app = Flask(__name__)

@app.route('/scrape', methods=['POST'])
def scrape_site():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    try:
        sitemap_urls = crawl_sitemap(url)
        if not sitemap_urls:
            return jsonify({"error": "Could not retrieve any URLs from the sitemap"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    # Aquí iría tu lógica para procesar las URLs obtenidas del sitemap
    # Por simplicidad, solo devolvemos las URLs encontradas
    return jsonify({"sitemap_urls": sitemap_urls})

if __name__ == "__main__":
    app.run(debug=True)
