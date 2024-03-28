import pytest
from content_extractor import SEOContentAnalyzer  # Asegúrate de que la ruta sea correcta

@pytest.mark.asyncio
async def test_fetch_content():
    url = "https://example.com"  # Sustituye con una URL real para probar
    analyzer = SEOContentAnalyzer(url)
    content = await analyzer.analyze_content()
    assert content is not None, "El contenido extraído no debe ser None."
    assert 'title' in content, "Debería extraerse el título."

@pytest.mark.asyncio
async def test_keywords_extraction():
    url = "https://example.com"  # Sustituye con una URL real para probar
    analyzer = SEOContentAnalyzer(url)
    content = await analyzer.analyze_content()
    assert len(content['keywords']) > 0, "Deberían extraerse las palabras clave."
