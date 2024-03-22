import requests
from xml.etree import ElementTree
import time

def fetch_robots_txt(domain_url):
    """Fetch the robots.txt file from the given domain URL."""
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
    return None

def fetch_sitemap_content(sitemap_url):
    """Fetch the sitemap content from the given URL."""
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

if __name__ == "__main__":
    import sys
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    })

    if len(sys.argv) > 1:
        domain_url = sys.argv[1]
        robots_txt_content = fetch_robots_txt(domain_url)
        if robots_txt_content:
            sitemap_url = extract_sitemap_url(robots_txt_content)
            if sitemap_url:
                print(f"Sitemap found at: {sitemap_url}")
                sitemap_content = fetch_sitemap_content(sitemap_url)
                if sitemap_content:
                    urls = extract_urls_from_sitemap(sitemap_content)
                    print(f"URLs found in the sitemap: {urls}")
                else:
                    print("No URLs found in the sitemap.")
            else:
                print("No sitemap found in robots.txt.")
        else:
            print("Failed to fetch robots.txt.")
    else:
        print("Please provide a domain URL as an argument.")
