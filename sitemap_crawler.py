import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import json
from urllib.parse import urlparse
from datetime import datetime
import time

def fetch_and_parse(url, visited, headers=None):
    # Si ya visitamos la URL, retornamos None para evitar procesamiento adicional
    if url in visited:
        print(f"URL already processed: {url}")
        return None
    visited.add(url)

    headers = headers or {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title').text.strip() if soup.find('title') else 'No title found'
        h1_tags = [h1.text.strip() for h1 in soup.find_all('h1')]
        data = {"url": url, "title": title, "h1_tags": h1_tags}
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching and parsing {url}: {e}")
        return None

def fetch_sitemap(url, headers=None):
    headers = headers or {}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching sitemap: {e}")
        return None

def parse_sitemap(sitemap_content, sitemap_urls, visited, headers=None):
    try:
        root = ET.fromstring(sitemap_content)
        namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        for sitemap in root.findall("sitemap:sitemap", namespace):
            loc = sitemap.find("sitemap:loc", namespace).text
            if loc not in sitemap_urls:
                sitemap_urls.add(loc)
                sitemap_content = fetch_sitemap(loc, headers=headers)
                if sitemap_content:
                    parse_sitemap(sitemap_content, sitemap_urls, visited, headers=headers)
        for url in root.findall("sitemap:url", namespace):
            loc = url.find("sitemap:loc", namespace).text
            sitemap_urls.add(loc)
    except ET.ParseError as e:
        print(f"Error parsing sitemap XML: {e}")
    return sitemap_urls

def crawl_sitemap(sitemap_url, headers=None):
    visited = set()  # Set para rastrear las URLs ya procesadas
    sitemap_urls = set()
    sitemap_content = fetch_sitemap(sitemap_url, headers=headers)
    if sitemap_content:
        sitemap_urls = parse_sitemap(sitemap_content, sitemap_urls, visited, headers=headers)
        results = []
        for url in sitemap_urls:
            print(f"Processing URL: {url}")
            time.sleep(1)  # Respeta las políticas de cortesía con un pequeño delay
            result = fetch_and_parse(url, visited, headers=headers)
            if result:
                results.append(result)
                # Guarda los datos de cada URL individualmente
                domain = urlparse(url).netloc
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_file = f"data_{domain}_{timestamp}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
        # Además, podrías querer guardar un resumen de todos los resultados
        print(f"Data saved for {len(results)} URLs.")
    else:
        print("Failed to retrieve sitemap.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sitemap_url = sys.argv[1]
        crawl_sitemap(sitemap_url)
    else:
        print("Usage: python sitemap_crawler.py <sitemap_url>")
