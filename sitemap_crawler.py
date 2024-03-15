import requests
import xml.etree.ElementTree as ET

def fetch_sitemap(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.content
    except requests.exceptions.RequestException:
        return None

def parse_sitemap(content):
    sitemap_urls = []
    try:
        root = ET.fromstring(content)
        for url in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
            loc = url.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
            if loc is not None:
                sitemap_urls.append(loc.text)
        return sitemap_urls
    except ET.ParseError:
        return None

def crawl_sitemap(sitemap_url):
    content = fetch_sitemap(sitemap_url)
    if content:
        return parse_sitemap(content)
    else:
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sitemap_url = sys.argv[1]
        urls = crawl_sitemap(sitemap_url)
        if urls:
            print(f"Sitemap URLs: {urls}")
        else:
            print("Failed to retrieve or parse sitemap.")
    else:
        print("Please provide a sitemap URL as an argument.")
