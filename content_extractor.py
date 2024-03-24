import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from collections import Counter

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
        
        # Filtra los enlaces no útiles
        valid_links = [a['href'] for a in soup.find_all('a', href=True) if not a['href'].startswith(('#', 'javascript:void(0)'))]

        # Extrae elementos relevantes para SEO
        title = soup.title.text.strip() if soup.title else ''
        meta_description = soup.find("meta", attrs={"name": "description"})
        meta_description_content = meta_description["content"].strip() if meta_description and "content" in meta_description.attrs else ''
        h1_tags = [h1.text.strip() for h1 in soup.find_all('h1')]
        h2_tags = [h2.text.strip() for h2 in soup.find_all('h2')]
        h3_tags = [h3.text.strip() for h3 in soup.find_all('h3')]
        bold_tags = [bold.text.strip() for bold in soup.find_all(['b', 'strong'])]
        images = [{'src': img['src'], 'alt': img.get('alt', '').strip()} for img in soup.find_all('img') if img.get('src')]
        
        # Análisis de texto completo del cuerpo
        paragraphs = [p.text for p in soup.find_all('p')]
        clean_paragraphs = [para.strip() for para in paragraphs if para.strip() != '']
        body_text = ' '.join(clean_paragraphs)
        blob = TextBlob(body_text)
        
        # Extracción de sustantivos como keywords potenciales
        keywords = [word for word, tag in blob.tags if tag.startswith('N')]
        keyword_freq = Counter(keywords).most_common(20)
        
        noun_phrases = blob.noun_phrases
        entity_freq = Counter(noun_phrases).most_common(20)
        
        return {
            'title': title,
            'meta_description': meta_description_content,
            'h1_tags': h1_tags,
            'h2_tags': h2_tags,
            'h3_tags': h3_tags,
            'bold_tags': bold_tags,
            'images': images,
            'links': valid_links,
            'keywords': keyword_freq,
            'entities': entity_freq,
        }

# Aceptar URL de entrada del usuario
input_url = input("Por favor, introduce la URL que deseas analizar: ")
analyzer = SEOContentAnalyzer(input_url)
seo_report = analyzer.analyze_content()

# Imprimir el reporte SEO
print(seo_report)
