import requests
from bs4 import BeautifulSoup
from lxml import html

class ContentExtractor:
    def __init__(self, url):
        self.url = url

    def fetch_page_content(self):
        """Fetches the content of a page given its URL."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        try:
            response = requests.get(self.url, headers=headers)
            if response.status_code == 200:
                return response.content
            else:
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching the URL: {e}")
            return None

    def extract_links_and_headings(self, content):
        """Extracts all links and headings (h1, h2, h3) from the page content."""
        webpage = html.fromstring(content)
        soup = BeautifulSoup(content, "lxml")

        # Extract links
        links = webpage.xpath('//a/@href')

        # Extract headings
        headings = {'h1': [], 'h2': [], 'h3': []}
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            headings[heading.name].append(heading.text.strip())

        return links, headings

    def extract_content(self):
        """Main method to extract content, links, and headings from the URL."""
        content = self.fetch_page_content()
        if content:
            links, headings = self.extract_links_and_headings(content)
            soup = BeautifulSoup(content, 'lxml')
            body_text = soup.get_text(separator=' ', strip=True)  # Extract all text from the body
            return {
                'links': links,
                'headings': headings,
                'body_text': body_text
            }
        else:
            return None
