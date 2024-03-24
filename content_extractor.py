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
        
        valid_links = [a['href'] for a in soup.find_all('a', href=True) if not a['href'].startswith(('#', 'javascript:void(0)'))]

        title = soup.title.text.strip() if soup.title else ''
        meta_description = soup.find("meta", attrs={"name": "description"})
        meta_description_content = meta_description["content"].strip() if meta_description and "content" in meta_description.attrs else ''
        h1_tags = [h1.text.strip() for h1 in soup.find_all('h1')]
        h2_tags = [h2.text.strip() for h2 in soup.find_all('h2')]
        
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
            'h1_tags': h1_tags,
            'h2_tags': h2_tags,
            'links': valid_links,
            'keywords': keyword_freq,
            'entities': entity_freq,
        }
