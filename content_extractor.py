import requests
from lxml import html
from bs4 import BeautifulSoup
import sys

class ContentExtractor:
    def __init__(self, url):
        self.url = url

    def fetch_page_content(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        try:
            response = requests.get(self.url, headers=headers)
            if response.status_code == 200:
                return response.content
            else:
                print(f"Failed to fetch URL: HTTP Status Code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching the URL: {self.url}\n{e}")
        return None

    def extract_links_and_headings(self, content):
        webpage = html.fromstring(content)
        soup = BeautifulSoup(content, "lxml")

        # Extract links
        links = webpage.xpath('//a/@href')

        # Extract headings
        headings = {'h1': [], 'h2': [], 'h3': []}
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            headings[heading.name].append(heading.text.strip())

        return links, headings

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
        extractor = ContentExtractor(url)
        page_content = extractor.fetch_page_content()
        if page_content:
            links, headings = extractor.extract_links_and_headings(page_content)
            print("Links obtenidos exitosamente:")
            for link in links:
                print(link)
            print("\nEncabezados extraídos exitosamente:")
            for tag, texts in headings.items():
                for text in texts:
                    print(f'{tag}: {text}')
    else:
        print("Por favor, proporciona una URL como argumento.")
