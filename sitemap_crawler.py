import requests
from xml.etree import ElementTree

class SitemapExtractor:
    def __init__(self, domain_url):
        self.domain_url = domain_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0'
        })

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
            sitemap_tags = root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap')
            url_tags = root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url')
            for sitemap in sitemap_tags:
                loc = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc').text
                nested_content = self.fetch_sitemap_content(loc)
                urls.extend(self.extract_urls_from_sitemap(nested_content))
            for url in url_tags:
                loc = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc').text
                urls.append(loc)
        except ElementTree.ParseError as e:
            print(f"Error parsing sitemap: {e}")
        return urls

    def crawl_sitemap(self):
        urls_found = []
        sitemap_urls = [
            f"{self.domain_url}/sitemap.xml",
            f"{self.domain_url}/sitemap_index.xml",
            f"{self.domain_url}/wp-sitemap.xml"  # Agregado soporte para WordPress
        ]
        for sitemap_url in sitemap_urls:
            sitemap_content = self.fetch_sitemap_content(sitemap_url)
            if sitemap_content:
                urls_found.extend(self.extract_urls_from_sitemap(sitemap_content))
                break  # Detiene después de encontrar el primer sitemap válido
        return urls_found
