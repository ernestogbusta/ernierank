# test_analyze_content.py
from content_extractor import SEOContentAnalyzer

# Usa una URL de ejemplo para el análisis. Cambia esto por la URL que desees probar.
test_url = 'https://aulacm.com/guia-hacer-auditoria-seo/'

analyzer = SEOContentAnalyzer(test_url)
results = analyzer.analyze_content()
print(results)
