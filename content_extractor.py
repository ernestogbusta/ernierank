# content_extractor.py

import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from collections import Counter

class SEOContentAnalyzer:
    def __init__(self, url):
        self.url = url

    def fetch_content(self):
        """Try to fetch the content from specified URL and handle connection errors."""
        try:
            response = requests.get(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching page: {e}")
            return None

    def analyze_content(self):
        """Analyze the HTML content of a webpage for SEO, excluding comments."""
        html_content = self.fetch_content()
        if not html_content:
            return None

        soup = BeautifulSoup(html_content, 'html.parser')

        # Extracting key elements for SEO
        title = soup.title.text.strip() if soup.title else ''
        meta_description = soup.find("meta", attrs={"name": "description"})
        meta_description_content = meta_description["content"].strip() if meta_description and "content" in meta_description.attrs else ''
        canonical_link = soup.find("link", rel="canonical")
        canonical_href = canonical_link['href'] if canonical_link else ''

        # Preparing and analyzing body text for keyword and entity extraction
        body_text = ' '.join(p.text for p in soup.find_all('p'))
        blob = TextBlob(body_text)

        keywords = [word for word, tag in blob.tags if tag.startswith('N')]
        keyword_freq = Counter(keywords).most_common(20)
        noun_phrases = blob.noun_phrases
        entity_freq = Counter(noun_phrases).most_common(20)

        return {
            'title': title,
            'meta_description': meta_description_content,
            'canonical_link': canonical_href,
            'keywords': [keyword for keyword, _ in keyword_freq],
            'entities': [entity for entity, _ in entity_freq],
        }
