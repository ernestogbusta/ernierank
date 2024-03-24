import requests
from bs4 import BeautifulSoup
import spacy
from collections import Counter

nlp = spacy.load("es_core_news_md")  # Asegúrate de que este modelo esté descargado.

class SEOContentAnalyzer:
    def __init__(self, url):
        self.url = url

    def fetch_content(self):
        try:
            response = requests.get(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error al obtener la página: {e}")
            return None

    def analyze_content(self):
        html_content = self.fetch_content()
        if not html_content:
            return None
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        doc = nlp(text)
        
        keywords = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
        keyword_freq = Counter(keywords).most_common(20)
        
        entities = [entity.text for entity in doc.ents]
        entity_freq = Counter(entities).most_common(20)
        
        # Implementa tus métodos de análisis aquí
        
        return {
            'keywords': keyword_freq,
            'entities': entity_freq,
            # Añade más datos según tu análisis
        }
