import pytest
from sitemap_crawler import SitemapExtractor
from content_extractor import SEOContentAnalyzer

@pytest.mark.asyncio
async def test_fetch_content():
    # Aquí se prueba la funcionalidad de SEOContentAnalyzer de manera asincrónica
    analyzer = SEOContentAnalyzer("https://example.com")
    content = await analyzer.analyze_content()
    assert content is not None, "El contenido extraído no debería ser None."

@pytest.mark.asyncio
async def test_sitemap_extraction():
    # Asegúrate de cambiar "https://aulacm.com/sitemap_index.xml" por la URL real que quieras probar dinámicamente
    extractor = SitemapExtractor("https://aulacm.com/sitemap_index.xml")
    urls = await extractor.crawl_sitemap()
    assert len(urls) > 0, "Debería encontrar al menos una URL en el sitemap."
