import requests
from bs4 import BeautifulSoup
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_lg")  # Considera usar un modelo más grande para un análisis más profundo

class SEOContentAnalyzer:
    def __init__(self, url):
        self.url = url

    def fetch_content(self):
        """Obtiene el contenido HTML de la URL."""
        try:
            response = requests.get(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error al obtener la página: {e}")
            return None

    def analyze_content(self, html_content):
        """Realiza un análisis detallado del contenido HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        doc = nlp(text)
        
        # Extracción de palabras clave
        keywords = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
        keyword_freq = Counter(keywords).most_common(20)
        
        # Identificación de entidades para determinar topics
        entities = [entity.text for entity in doc.ents]
        entity_freq = Counter(entities).most_common(20)
        
        # Intención de búsqueda basada en el análisis del contenido
        search_intent = self.determine_search_intent(doc)
        
        # Análisis de la estructura semántica del contenido
        semantic_structure = self.analyze_semantic_structure(soup)
        
        return {
            'keywords': keyword_freq,
            'entities': entity_freq,
            'search_intent': search_intent,
            'semantic_structure': semantic_structure,
        }

    def determine_search_intent(self, doc):
        """Determina la intención de búsqueda del contenido."""
        # Implementa tu lógica aquí, basada en el análisis de las palabras clave y entidades
        return "Informativa/Transaccional/Navegacional"  # Ejemplo simplificado

    def analyze_semantic_structure(self, soup):
        """Analiza la jerarquía de encabezados y la estructura semántica."""
        headings = {f'h{i}': len(soup.find_all(f'h{i}')) for i in range(1, 7)}
        return headings

if __name__ == "__main__":
    url = "https://example.com"
    analyzer = SEOContentAnalyzer(url)
    html_content = analyzer.fetch_content()
    if html_content:
        analysis_results = analyzer.analyze_content(html_content)
        print(analysis_results)
    else:
        print("No se pudo analizar el contenido.")
