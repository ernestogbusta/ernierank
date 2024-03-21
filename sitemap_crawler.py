import requests
from xml.etree import ElementTree

def fetch_sitemap_content(sitemap_url):
    """Fetch the sitemap content from the given URL."""
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()  # This will raise an error for status codes >= 400
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
        # Extract URLs from sitemap index file
        for sitemap in root.findall('sitemap:sitemap', namespaces):
            loc = sitemap.find('sitemap:loc', namespaces).text
            urls.extend(crawl_sitemap(loc))  # Recursively process nested sitemaps
        # Extract URLs from a regular sitemap
        for url in root.findall('sitemap:url', namespaces):
            loc = url.find('sitemap:loc', namespaces).text
            urls.append(loc)
    except ElementTree.ParseError as e:
        print(f"Error parsing the sitemap XML: {e}")
    return urls

def crawl_sitemap(sitemap_url):
    """Crawl the sitemap and return all found URLs."""
    sitemap_content = fetch_sitemap_content(sitemap_url)
    if sitemap_content:
        urls = extract_urls_from_sitemap(sitemap_content)
        return urls
    else:
        return []

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sitemap_url = sys.argv[1]
        urls = crawl_sitemap(sitemap_url)
        print(f"Found URLs: {urls}")
    else:
        print("Please provide a sitemap URL as an argument.")
