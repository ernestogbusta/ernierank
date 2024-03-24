import requests
from xml.etree import ElementTree
import sys

# Inicialización de una sesión de requests para reutilizar conexiones TCP.
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
            return response.content  # Usar .content para evitar problemas de codificación
        except requests.RequestException as e:
            print(f"Error al obtener el sitemap: {e}")
            return None

    def extract_urls_from_sitemap(sitemap_content):
        namespaces = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = []
        try:
            root = ElementTree.fromstring(sitemap_content)
            # Soporte tanto para sitemaps de URLs como índices de sitemaps.
            for sitemap in root.findall('sitemap:sitemap', namespaces) or root.findall('sitemap:url', namespaces):
                loc = sitemap.find('sitemap:loc', namespaces).text
                if 'sitemap' in loc:
                    urls.extend(extract_urls_from_sitemap(fetch_sitemap_content(loc)))
                else:
                    urls.append(loc)
        except ElementTree.ParseError as e:
            print(f"Error al parsear el sitemap: {e}")
        return urls

    urls_found = []
    for sitemap_url in common_sitemap_urls:
        sitemap_content = fetch_sitemap_content(sitemap_url)
        if sitemap_content:
            urls_found.extend(extract_urls_from_sitemap(sitemap_content))

    return urls_found

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
