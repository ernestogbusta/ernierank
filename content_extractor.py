import requests
from bs4 import BeautifulSoup, Comment
from textblob import TextBlob
from collections import Counter

class SEOContentAnalyzer:
    def __init__(self, url):
        self.url = url

    def fetch_content(self):
        """Fetches the content of the URL."""
        try:
            response = requests.get(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching page: {e}")
            return None

    def analyze_content(self):
        """Analyzes the HTML content for SEO elements."""
        html_content = self.fetch_content()
        if not html_content:
            return None

        soup = BeautifulSoup(html_content, 'html.parser')
        # Removal of comment sections
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        title = soup.title.text.strip() if soup.title else ''
        meta_description = soup.find("meta", attrs={"name": "description"})["content"].strip() if soup.find("meta", attrs={"name": "description"}) else ''
        canonical_link = soup.find("link", rel="canonical")["href"] if soup.find("link", rel="canonical") else ''

        # Simplification of content analysis
        paragraphs = ' '.join(p.text for p in soup.find_all('p'))
        blob = TextBlob(paragraphs)

        keywords = [word for word, tag in blob.tags if tag.startswith('N')]
        keyword_freq = Counter(keywords).most_common(20)
        noun_phrases = blob.noun_phrases
        entity_freq = Counter(noun_phrases).most_common(20)

        return {
            'title': title,
            'meta_description': meta_description,
            'canonical_link': canonical_link,
            'keywords': [keyword for keyword, _ in keyword_freq],
            'entities': [entity for entity, _ in entity_freq],
        }
