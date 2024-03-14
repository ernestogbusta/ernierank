import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from datetime import datetime
import time

def fetch_and_parse(url, visited, headers=None):
    if url in visited:
        return None
    visited.add(url)

    headers = headers or {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Considerar extracción de más datos si es necesario
        title = soup.find('title').text.strip() if soup.find('title') else 'No title found'
        return {"url": url, "title": title}
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def fetch_sitemap(url, headers=None):
    headers = headers or {}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None

def parse_sitemap(sitemap_content, sitemap_urls, visited, headers=None):
    try:
        root = ET.fromstring(sitemap_content)
        namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        # Asegurarse de manejar tanto sitemaps como urls
        sitemaps = root.findall("sitemap:sitemap", namespace)
        urls = root.findall("sitemap:url", namespace)
        for sitemap in sitemaps:
            loc = sitemap.find("sitemap:loc", namespace).text
            if loc not in sitemap_urls:
                sitemap_urls.add(loc)
                content = fetch_sitemap(loc, headers=headers)
                if content:
                    parse_sitemap(content, sitemap_urls, visited, headers=headers)
        for url in urls:
            loc = url.find("sitemap:loc", namespace).text
            sitemap_urls.add(loc)
    except ET.ParseError as e:
        print(f"Error: {e}")
    return sitemap_urls

def crawl_sitemap(sitemap_url, headers=None):
    visited, sitemap_urls = set(), set()
    sitemap_content = fetch_sitemap(sitemap_url, headers=headers)
    if sitemap_content:
        sitemap_urls = parse_sitemap(sitemap_content, sitemap_urls, visited, headers=headers)
        for url in sitemap_urls:
            time.sleep(1)  # Política de cortesía
            result = fetch_and_parse(url, visited, headers=headers)
            if result:
                # Considerar implementación para el manejo de resultados
                print(f"Data processed for URL: {url}")
    else:
        print("Failed to retrieve sitemap.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        crawl_sitemap(sys.argv[1])
    else:
        print("Provide a sitemap URL as an argument.")
