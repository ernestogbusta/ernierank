import requests
from lxml import html
import sys

class LinkExtractor:
    def __init__(self, url):
        self.url = url
        self.links = []

    def fetch_and_extract_links(self):
        try:
            response = requests.get(self.url)
            if response.status_code == 200:
                webpage = html.fromstring(response.content)
                self.links = webpage.xpath('//a/@href')
                return True
            else:
                print(f"Failed to fetch URL: HTTP Status Code {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Error fetching the URL: {self.url}\n{e}")
            return False

    def print_links(self):
        if self.links:
            print("Links obtenidos exitosamente:")
            for link in self.links:
                print(link)
        else:
            print("No se encontraron links o no fue posible extraerlos.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
        extractor = LinkExtractor(url)
        if extractor.fetch_and_extract_links():
            extractor.print_links()
    else:
        print("Por favor, proporciona una URL como argumento.")
