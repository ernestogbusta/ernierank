import requests
from xml.etree import ElementTree
import sys

# Inicializamos la sesión aquí para que sea reutilizable.
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
})

def crawl_sitemap(domain_url):
    """Encapsula la lógica completa para recuperar URLs desde un sitemap."""
    def fetch_robots_txt():
        """Fetch the robots.txt file."""
        try:
            response = session.get(f"{domain_url}/robots.txt")
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching robots.txt: {e}")
            return None

    def extract_sitemap_url(robots_txt_content):
        """Extract sitemap URL from the robots.txt content."""
        for line in robots_txt_content.splitlines():
            if line.startswith('Sitemap:'):
                return line.split(': ')[1]
        # Intenta ubicaciones comunes de sitemap si no se encuentra en robots.txt
        for common_sitemap_url in ['/sitemap.xml', '/sitemap_index.xml']:
            try:
                response = session.head(f"{domain_url}{common_sitemap_url}")
                if response.status_code == 200 or (300 <= response.status_code < 400):
                    # Retorna la URL final después de redirecciones
                    return response.url
            except requests.RequestException:
                continue  # Intenta con la siguiente URL común si esta falla
        return None

    def fetch_sitemap_content(sitemap_url):
        """Fetch the sitemap content."""
        try:
            response = session.get(sitemap_url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching the sitemap: {e}")
            return None

    def extract_urls_from_sitemap(sitemap_content):
        """Extract URLs from sitemap content."""
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
        except ElementTree.ParseError as e:
            print(f"Error parsing the sitemap XML: {e}")
        return urls

    robots_txt_content = fetch_robots_txt()
    sitemap_url = extract_sitemap_url(robots_txt_content) if robots_txt_content else None
    if not sitemap_url:
        print("Sitemap URL not found in robots.txt, trying common locations...")
        sitemap_url = extract_sitemap_url("")  # Intentionally empty to trigger common location checks
    if sitemap_url:
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
