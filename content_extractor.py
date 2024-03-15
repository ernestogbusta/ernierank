import requests
from bs4 import BeautifulSoup

class ContentExtractor:
    def __init__(self, url):
        self.url = url

    def fetch_page_content(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        try:
            response = requests.get(self.url, headers=headers)
            return response if response.status_code == 200 else None
        except requests.exceptions.RequestException:
            return None

    def extract_content(self):
        response = self.fetch_page_content()
        if response:
            soup = BeautifulSoup(response.content, 'html.parser')
            body_text = soup.get_text(separator=' ', strip=True)
            headings = {tag.name: [heading.get_text(strip=True) for heading in soup.find_all(tag.name)] for tag in ['h1', 'h2', 'h3']}
            links = [a['href'] for a in soup.find_all('a', href=True)]
            return {'body_text': body_text, 'headings': headings, 'links': links}
        else:
            return {'error': 'Failed to fetch or parse the URL content'}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        url = sys.argv[1]
        extractor = ContentExtractor(url)
        extracted_data = extractor.extract_content()
        print(extracted_data)
    else:
        print("Please provide a URL as an argument.")
