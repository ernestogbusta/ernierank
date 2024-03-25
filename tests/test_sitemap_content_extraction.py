# /Users/ernestogarciabustamante/ernierank/tests/test_sitemap_content_extraction.py
import unittest
import sys
import os

# Ajusta la ruta al directorio de tu aplicación
ruta_directorio_aplicacion = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ruta_directorio_aplicacion)

from sitemap_crawler import SitemapExtractor
from content_extractor import SEOContentAnalyzer

class TestSitemapContentExtraction(unittest.TestCase):
    def test_sitemap_extraction(self):
        extractor = SitemapExtractor("https://aulacm.com")
        urls = extractor.crawl_sitemap()
        self.assertGreater(len(urls), 0, "Debería encontrar al menos una URL en el sitemap.")

    def test_home_content_extraction(self):
        analyzer = SEOContentAnalyzer("https://aulacm.com")
        content = analyzer.analyze_content()
        self.assertNotEqual(content['title'], '', "El título extraído no debería estar vacío.")
        # Esta línea es modificada para adaptarse a la estructura actual de la respuesta de `analyze_content()`
        self.assertTrue(any(tag in content['body_text'].lower() for tag in ['<html', '<div', '<p']), "El contenido extraído debería incluir etiquetas HTML básicas.")

if __name__ == '__main__':
    unittest.main()
