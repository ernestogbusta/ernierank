from aiohttp import web
from content_extractor import SEOContentAnalyzer
from sitemap_crawler import SitemapExtractor

async def handle_content_analysis(request):
    data = await request.json()
    url = data.get('url')
    analyzer = SEOContentAnalyzer(url)
    content = await analyzer.analyze_content()
    return web.json_response(content)

async def handle_sitemap_crawling(request):
    params = request.rel_url.query
    url = params.get('sitemap_url')
    extractor = SitemapExtractor(url)
    urls = await extractor.crawl_sitemap()
    return web.json_response({"urls": urls})

app = web.Application()
app.add_routes([web.post('/analyze', handle_content_analysis),
                web.get('/crawl-sitemap', handle_sitemap_crawling)])

if __name__ == '__main__':
    web.run_app(app, port=8080)
