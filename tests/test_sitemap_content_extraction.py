# tests/test_sitemap_content_extraction.py

import unittest
import sys
import os

# Ajusta la ruta al directorio de tu aplicación para poder importar tus módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sitemap_crawler import SitemapExtractor
from content_extractor import SEOContentAnalyzer

class TestSitemapContentExtraction(unittest.TestCase):
    def test_sitemap_extraction(self):
        # Reemplaza "https://example.com" con la URL real que quieras probar
        extractor = SitemapExtractor("https://ernestogbustamante.com/")
        urls = extractor.crawl_sitemap()
        self.assertGreater(len(urls), 0, "Should find at least one URL in the sitemap.")

    def test_home_content_extraction(self):
        # Reemplaza "https://example.com" con la URL real que quieras probar
        analyzer = SEOContentAnalyzer("https://ernestogbustamante.com/")
        content = analyzer.analyze_content()
        self.assertIsNotNone(content, "The extracted content should not be None.")
        # Asegúrate de que el título extraído no esté vacío como una prueba básica
        self.assertNotEqual(content.get('title', ''), '', "The extracted title should not be empty.")

if __name__ == '__main__':
    unittest.main()
