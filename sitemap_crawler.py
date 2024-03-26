import requests
from xml.etree import ElementTree

class SitemapExtractor:
    def __init__(self, domain_url):
        self.domain_url = domain_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; YourBotName/1.0; +http://www.yourwebsite.com/bot.html)'
        })

    def fetch_sitemap_content(self, sitemap_url):
        """Fetch the sitemap content from a given URL."""
        try:
            response = self.session.get(sitemap_url)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            print(f"Error fetching sitemap from {sitemap_url}: {e}")
            return None

    def extract_urls_from_sitemap(self, sitemap_content):
        """Extract URLs from the sitemap content, handling both index and standard sitemaps."""
        urls = []
        try:
            root = ElementTree.fromstring(sitemap_content)
            # Check if it's an index sitemap
            if root.tag.endswith('sitemapindex'):
                for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                    loc = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc').text
                    nested_content = self.fetch_sitemap_content(loc)
                    if nested_content:
                        urls.extend(self.extract_urls_from_sitemap(nested_content))
            else:
                for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                    loc = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc').text
                    urls.append(loc)
        except ElementTree.ParseError as e:
            print(f"Error parsing sitemap: {e}")
        return urls

    def crawl_sitemap(self):
        """Crawl the sitemaps starting from the domain URL."""
        urls_found = []
        sitemap_endpoints = ["sitemap.xml", "sitemap_index.xml"]
        for endpoint in sitemap_endpoints:
            sitemap_url = f"{self.domain_url.rstrip('/')}/{endpoint}"
            sitemap_content = self.fetch_sitemap_content(sitemap_url)
            if sitemap_content:
                urls_found.extend(self.extract_urls_from_sitemap(sitemap_content))
        return urls_found

# Example usage:
if __name__ == "__main__":
    domain_to_crawl = "https://aulacm.com"
    extractor = SitemapExtractor(domain_to_crawl)
    urls = extractor.crawl_sitemap()
    print(f"Found {len(urls)} URLs in the sitemap of {domain_to_crawl}")
    for url in urls:
        print(url)
