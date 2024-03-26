import requests
from xml.etree import ElementTree

class SitemapExtractor:
    def __init__(self, domain_url):
        self.domain_url = domain_url
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})

    def fetch_sitemap_content(self, sitemap_url):
        try:
            response = self.session.get(sitemap_url)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            print(f"Error fetching sitemap from {sitemap_url}: {e}")
            return None

    def extract_urls_from_sitemap(self, sitemap_content):
        urls = []
        try:
            root = ElementTree.fromstring(sitemap_content)
            for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc').text
                urls.append(loc)
        except ElementTree.ParseError as e:
            print(f"Error parsing sitemap: {e}")
        return urls

    def crawl_sitemap(self):
        urls_found = []
        sitemap_urls = [f"{self.domain_url}/sitemap.xml", f"{self.domain_url}/sitemap_index.xml"]
        for sitemap_url in sitemap_urls:
            sitemap_content = self.fetch_sitemap_content(sitemap_url)
            if sitemap_content:
                urls_found.extend(self.extract_urls_from_sitemap(sitemap_content))
        return urls_found
