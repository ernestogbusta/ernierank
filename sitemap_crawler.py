import requests
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

def find_sitemap_url(domain_url):
    """Find the sitemap URL by parsing the robots.txt file."""
    parsed_url = urlparse(domain_url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    robots_txt_url = urljoin(base_url, "/robots.txt")
    
    try:
        response = requests.get(robots_txt_url)
        response.raise_for_status()
        for line in response.text.splitlines():
            if line.startswith("Sitemap:"):
                return line.split(":", 1)[1].strip()
    except requests.RequestException as e:
        print(f"Error fetching the robots.txt: {e}")
    
    return None

def fetch_sitemap_content(sitemap_url):
    """Fetch the sitemap content from the given URL."""
    try:
        response = requests.get(sitemap_url)
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
            urls.extend(crawl_sitemap(loc))
        for url in root.findall('sitemap:url', namespaces):
            loc = url.find('sitemap:loc', namespaces).text
            urls.append(loc)
    except ElementTree.ParseError as e:
        print(f"Error parsing the sitemap XML: {e}")
    return urls

def crawl_sitemap(domain_url):
    """Crawl the sitemap and return all found URLs."""
    sitemap_url = find_sitemap_url(domain_url)
    if sitemap_url:
        sitemap_content = fetch_sitemap_content(sitemap_url)
        if sitemap_content:
            return extract_urls_from_sitemap(sitemap_content)
        else:
            print("Failed to fetch the sitemap content.")
    else:
        print("Sitemap URL not found in robots.txt.")
    return []

if __name__ == "__main__":
    if len(sys.argv) > 1:
        domain_url = sys.argv[1]
        urls = crawl_sitemap(domain_url)
        print(f"Found URLs: {urls}")
    else:
        print("Please provide a domain URL as an argument.")
