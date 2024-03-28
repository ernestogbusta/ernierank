# tests/test_sitemap_content_extraction.py

import pytest
import sys
import os

# Ajusta la ruta al directorio de tu aplicación para poder importar tus módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sitemap_crawler import SitemapExtractor
from content_extractor import SEOContentAnalyzer

@pytest.mark.asyncio
async def test_sitemap_extraction():
    # Reemplaza "https://example.com" con la URL real que quieras probar
    extractor = SitemapExtractor("https://aulacm.com/sitemap_index.xml")
    urls = await extractor.crawl_sitemap()
    assert len(urls) > 0, "Should find at least one URL in the sitemap."

@pytest.mark.asyncio
async def test_home_content_extraction():
    # Reemplaza "https://example.com" con la URL real que quieras probar
    analyzer = SEOContentAnalyzer("https://aulacm.com/")
    content = await analyzer.analyze_content()
    assert content is not None, "The extracted content should not be None."
    # Asegúrate de que el título extraído no esté vacío como una prueba básica
    assert content.get('title', '') != '', "The extracted title should not be empty."
