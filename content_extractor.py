import requests
from bs4 import BeautifulSoup, Comment
from textblob import TextBlob
from collections import Counter

class SEOContentAnalyzer:
    def __init__(self, url):
        self.url = url

    def fetch_content(self):
        """Intenta obtener el contenido de la URL especificada y maneja errores de conexión."""
        try:
            response = requests.get(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error al obtener la página: {e}")
            return None

    def analyze_content(self):
        """Analiza el contenido HTML de una página web para SEO, excluyendo comentarios."""
        html_content = self.fetch_content()
        if not html_content:
            return None

        soup = BeautifulSoup(html_content, 'html.parser')

        # Eliminación de secciones de comentarios antes del análisis
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        [comment.extract() for comment in comments]
        # Identificar y eliminar secciones comunes de comentarios por ID o clases
        comment_selectors = ['div#comments', 'div.comments', '.comments-area', 'section.comments']
        for selector in comment_selectors:
            for comment_section in soup.select(selector):
                comment_section.decompose()

        # Extracción de enlaces válidos
        valid_links = [a['href'] for a in soup.find_all('a', href=True) if not a['href'].startswith(('#', 'javascript:void(0)', 'mailto:', 'tel:'))]

        # Extracción de elementos clave para SEO
        title = soup.title.text.strip() if soup.title else ''
        meta_description = soup.find("meta", attrs={"name": "description"})
        meta_description_content = meta_description["content"].strip() if meta_description and "content" in meta_description.attrs else ''
        canonical_link = soup.find("link", rel="canonical")
        canonical_href = canonical_link['href'] if canonical_link else ''
        images_alt = [img['alt'] for img in soup.find_all('img') if img.has_attr('alt')]
        h1_tags = [h1.text.strip() for h1 in soup.find_all('h1')]
        h2_tags = [h2.text.strip() for h2 in soup.find_all('h2')]

        # Preparación y análisis del texto del cuerpo para extracción de palabras clave y entidades
        paragraphs = [p.text for p in soup.find_all('p')]
        clean_paragraphs = [para.strip() for para in paragraphs if para.strip() != '']
        body_text = ' '.join(clean_paragraphs)
        blob = TextBlob(body_text)

        keywords = [word for word, tag in blob.tags if tag.startswith('N')]
        keyword_freq = Counter(keywords).most_common(20)
        noun_phrases = blob.noun_phrases
        entity_freq = Counter(noun_phrases).most_common(20)

        return {
            'title': title,
            'meta_description': meta_description_content,
            'canonical_link': canonical_href,
            'images_alt': images_alt,
            'h1_tags': h1_tags,
            'h2_tags': h2_tags,
            'links': valid_links,
            'body_text': body_text,  # Incluir el texto completo del cuerpo para visualización
            'keywords': [keyword for keyword, _ in keyword_freq],
            'entities': [entity for entity, _ in entity_freq],
        }
