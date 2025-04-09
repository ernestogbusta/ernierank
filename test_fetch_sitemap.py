import asyncio
import httpx
from urllib.parse import urlparse, urljoin
import xmltodict
import gzip
from bs4 import BeautifulSoup

async def fetch_sitemap(client, base_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        "Accept": "application/xml,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8"
    }

    base_domain = urlparse(base_url).scheme + "://" + urlparse(base_url).netloc
    sitemap_candidates = [
        f"{base_domain}/sitemap_index.xml",
        f"{base_domain}/sitemap.xml",
        f"{base_domain}/sitemap1.xml",
        f"{base_domain}/sitemap.xml.gz"
    ]

    all_urls = []

    # 1. Intentar sitemaps comunes
    for sitemap_url in sitemap_candidates:
        try:
            response = await client.get(sitemap_url, headers=headers, timeout=20)
            if response.status_code == 200:
                if sitemap_url.endswith('.gz'):
                    decompressed = gzip.decompress(response.content)
                    sitemap_contents = xmltodict.parse(decompressed)
                else:
                    sitemap_contents = xmltodict.parse(response.content)

                # Si es un índice de sitemaps
                if 'sitemapindex' in sitemap_contents:
                    sitemaps = sitemap_contents['sitemapindex'].get('sitemap', [])
                    sitemaps = sitemaps if isinstance(sitemaps, list) else [sitemaps]

                    for sitemap in sitemaps:
                        loc = sitemap.get('loc')
                        if loc:
                            urls = await fetch_individual_sitemap(client, loc)
                            all_urls.extend(urls)

                # Si es un urlset (sitemap simple)
                elif 'urlset' in sitemap_contents:
                    urls = sitemap_contents['urlset'].get('url', [])
                    urls = urls if isinstance(urls, list) else [urls]
                    all_urls.extend([url['loc'] for url in urls if 'loc' in url])

                if all_urls:
                    return all_urls  # ✅ Devolver sólo si hemos encontrado URLs
        except Exception as e:
            print(f"Error intentando acceder a {sitemap_url}: {e}")
            continue

    # 2. Intentar desde robots.txt
    try:
        sitemap_from_robots = await discover_sitemaps_from_robots_txt(client, base_domain)
        for sitemap_url in sitemap_from_robots:
            urls = await fetch_individual_sitemap(client, sitemap_url)
            all_urls.extend(urls)
        if all_urls:
            return all_urls
    except Exception as e:
        print(f"Error buscando sitemap en robots.txt: {e}")

    # 3. Último recurso: buscar en HTML principal
    try:
        response = await client.get(base_domain, headers=headers, timeout=20)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'sitemap' in href and href.endswith(('.xml', '.xml.gz')):
                    sitemap_url = urljoin(base_domain, href)
                    urls = await fetch_individual_sitemap(client, sitemap_url)
                    all_urls.extend(urls)
            if all_urls:
                return all_urls
    except Exception as e:
        print(f"Error buscando sitemap en HTML principal: {e}")

    return None  # ❌ No se encontró ningún sitemap

async def discover_sitemaps_from_robots_txt(client, base_domain):
    robots_url = f"{base_domain}/robots.txt"
    sitemaps = []
    try:
        response = await client.get(robots_url, timeout=10)
        if response.status_code == 200:
            for line in response.text.splitlines():
                if line.lower().startswith('sitemap:'):
                    sitemap_url = line.split(':', 1)[1].strip()
                    sitemaps.append(sitemap_url)
    except Exception as e:
        print(f"Error leyendo robots.txt en {robots_url}: {e}")
    return sitemaps

async def fetch_individual_sitemap(client, sitemap_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        "Accept": "application/xml,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8"
    }
    try:
        response = await client.get(sitemap_url, headers=headers, timeout=20)
        if sitemap_url.endswith('.gz'):
            decompressed = gzip.decompress(response.content)
            sitemap_contents = xmltodict.parse(decompressed)
        else:
            sitemap_contents = xmltodict.parse(response.content)

        if 'urlset' in sitemap_contents:
            return [url['loc'] for url in sitemap_contents['urlset']['url']]
        elif 'sitemapindex' in sitemap_contents:
            nested_sitemaps = sitemap_contents['sitemapindex'].get('sitemap', [])
            nested_sitemaps = nested_sitemaps if isinstance(nested_sitemaps, list) else [nested_sitemaps]
            all_urls = []
            for sitemap in nested_sitemaps:
                loc = sitemap['loc']
                urls = await fetch_individual_sitemap(client, loc)
                all_urls.extend(urls)
            return all_urls
    except Exception as e:
        print(f"Error procesando sitemap {sitemap_url}: {e}")
        return []

# 🔥 Función principal de prueba
async def main():
    async with httpx.AsyncClient() as client:
        domain = input("Introduce el dominio que quieres probar (por ejemplo https://ernestogbustamante.com): ").strip()
        urls = await fetch_sitemap(client, domain)
        if urls:
            print(f"\n✅ {len(urls)} URLs encontradas en el sitemap:\n")
            for url in urls:
                print(url)
        else:
            print("\n⚠️ No se encontraron URLs o no se pudo procesar el sitemap.")

if __name__ == "__main__":
    asyncio.run(main())
