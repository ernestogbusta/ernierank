from flask import Flask, request, jsonify
from content_extractor import SEOContentAnalyzer
from sitemap_crawler import SitemapExtractor

app = Flask(__name__)

@app.route('/scrape', methods=['POST'])
async def scrape():
    data = request.json
    url = data['url']

    # Inicializa tus clases con el URL
    content_analyzer = SEOContentAnalyzer(url)
    sitemap_extractor = SitemapExtractor(url)

    # Extrae y analiza el contenido
    content_analysis = await content_analyzer.analyze_content()
    sitemap_urls = await sitemap_extractor.crawl_sitemap()

    # Devuelve los resultados
    return jsonify({
        'content_analysis': content_analysis,
        'sitemap_urls': sitemap_urls
    })

if __name__ == '__main__':
    app.run(debug=True)
