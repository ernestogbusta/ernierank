# sitemap_crawler.py
import requests
from xml.etree import ElementTree

class SitemapExtractor:
    def __init__(self, domain_url):
        self.domain_url = domain_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        })

    def fetch_sitemap_content(self, sitemap_url):
        try:
            response = self.session.get(sitemap_url)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            print(f"Error al obtener el sitemap desde {sitemap_url}: {e}")
            return None

    def extract_urls_from_sitemap(self, sitemap_content):
        namespaces = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = []
        try:
            root = ElementTree.fromstring(sitemap_content)
            for sitemap in root.findall('sitemap:sitemap', namespaces) or root.findall('sitemap:url', namespaces):
                loc = sitemap.find('sitemap:loc', namespaces).text
                if 'sitemap' in loc:
                    nested_sitemap_content = self.fetch_sitemap_content(loc)
                    if nested_sitemap_content:
                        urls.extend(self.extract_urls_from_sitemap(nested_sitemap_content))
                else:
                    urls.append(loc)
        except ElementTree.ParseError as e:
            print(f"Error al parsear el sitemap: {e}")
        return urls

    def crawl_sitemap(self):
        common_sitemap_urls = [f"{self.domain_url}/sitemap.xml", f"{self.domain_url}/sitemap_index.xml"]
        urls_found = []
        for sitemap_url in common_sitemap_urls:
            sitemap_content = self.fetch_sitemap_content(sitemap_url)
            if sitemap_content:
                urls_found.extend(self.extract_urls_from_sitemap(sitemap_content))
        return urls_found
