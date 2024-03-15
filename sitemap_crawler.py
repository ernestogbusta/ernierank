import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import time

def fetch_and_parse(url, visited, headers=None):
    if url in visited:
        print(f"URL already processed: {url}")
        return None
    visited.add(url)

    headers = headers or {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None

def parse_sitemap(content, sitemap_urls, visited, headers=None):
    try:
        root = ET.fromstring(content)
        namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        for elem in root.findall(".//*"):
            if elem.tag == "{http://www.sitemaps.org/schemas/sitemap/0.9}loc":
                url = elem.text
                if url not in visited:
                    sitemap_urls.add(url)
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
    return sitemap_urls

def crawl_sitemap(sitemap_url, headers=None):
    visited, sitemap_urls = set(), set()
    content = fetch_and_parse(sitemap_url, visited, headers=headers)
    if content:
        sitemap_urls = parse_sitemap(content, sitemap_urls, visited, headers=headers)
        for url in sitemap_urls:
            time.sleep(1)  # Courtesy delay
            if url.endswith('.xml'):
                # Handle nested sitemap
                nested_content = fetch_and_parse(url, visited, headers=headers)
                if nested_content:
                    parse_sitemap(nested_content, sitemap_urls, visited, headers=headers)
            else:
                # Here you can handle the individual URL processing
                print(f"URL to process: {url}")
    else:
        print("Failed to retrieve or parse sitemap.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sitemap_url = sys.argv[1]
        crawl_sitemap(sitemap_url)
    else:
        print("Please provide a sitemap URL as an argument.")
