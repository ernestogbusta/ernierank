import requests
from bs4 import BeautifulSoup
import spacy
from collections import Counter

# Carga un modelo más ligero y configúralo para procesar solo los componentes necesarios
nlp = spacy.load("es_core_news_md", disable=["parser", "attribute_ruler", "lemmatizer"])
nlp.select_pipes(enable=["tok2vec", "ner"])

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
        
        # Extrae elementos relevantes para SEO
        title = soup.title.text if soup.title else ''
        meta_description = soup.find("meta", attrs={"name": "description"})
        meta_description_content = meta_description["content"] if meta_description and "content" in meta_description.attrs else ''
        h1_tags = [h1.text for h1 in soup.find_all('h1')]
        h2_tags = [h2.text for h2 in soup.find_all('h2')]
        
        # Análisis de texto del cuerpo para keywords y entidades
        body_text = ' '.join([p.text for p in soup.find_all('p')])
        doc = nlp(body_text)
        keywords = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
        keyword_freq = Counter(keywords).most_common(20)
        entities = [entity.text for entity in doc.ents]
        entity_freq = Counter(entities).most_common(20)
        
        return {
            'title': title,
            'meta_description': meta_description_content,
            'h1_tags': h1_tags,
            'h2_tags': h2_tags,
            'keywords': keyword_freq,
            'entities': entity_freq,
        }
