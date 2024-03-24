import requests
from xml.etree import ElementTree
import sys

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
})

def crawl_sitemap(domain_url):
    common_sitemap_urls = [f"{domain_url}/sitemap.xml", f"{domain_url}/sitemap_index.xml"]

    def fetch_sitemap_content(sitemap_url):
        try:
            response = session.get(sitemap_url)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return None

    def extract_urls_from_sitemap(sitemap_content):
        namespaces = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = []
        try:
            root = ElementTree.fromstring(sitemap_content)
            for sitemap in root.findall('sitemap:sitemap', namespaces):
                loc = sitemap.find('sitemap:loc', namespaces).text
                urls.extend(extract_urls_from_sitemap(fetch_sitemap_content(loc)))
            for url in root.findall('sitemap:url', namespaces):
                loc = url.find('sitemap:loc', namespaces).text
                urls.append(loc)
        except ElementTree.ParseError:
            pass
        return urls

    for sitemap_url in common_sitemap_urls:
        sitemap_content = fetch_sitemap_content(sitemap_url)
        if sitemap_content:
            return extract_urls_from_sitemap(sitemap_content)

    return []

if __name__ == "__main__":
    if len(sys.argv) > 1:
        domain_url = sys.argv[1]
        urls = crawl_sitemap(domain_url)
        if urls:
            print(f"URLs found in the sitemap: {len(urls)}")
            for url in urls:
                print(url)
        else:
            print("No URLs found in the sitemap.")
    else:
        print("Please provide a domain URL as an argument.")
