import asyncio
import httpx
from bs4 import BeautifulSoup

URL_INICIAL = "https://boanorte.es"
CONCURRENCY = 50

visited = set()
semaphore = asyncio.Semaphore(CONCURRENCY)

async def fetch(client, url):
    try:
        async with semaphore:
            response = await client.get(url, timeout=10)
            print(f"✅ {response.status_code} - {url}")
            if response.status_code == 200 and 'text/html' in response.headers.get("content-type", ""):
                return response.text
    except Exception as e:
        print(f"❌ Error en {url}: {e}")
    return None

async def crawl(client, url):
    if url in visited:
        return
    visited.add(url)

    html = await fetch(client, url)
    if not html:
        return

    soup = BeautifulSoup(html, "html.parser")
    internal_links = set()

    for link_tag in soup.find_all("a", href=True):
        href = link_tag['href']
        if href.startswith("/") or href.startswith(URL_INICIAL):
            absolute_url = href if href.startswith("http") else f"{URL_INICIAL}{href}"
            if absolute_url.startswith(URL_INICIAL) and absolute_url not in visited:
                internal_links.add(absolute_url)

    tasks = [crawl(client, link) for link in internal_links]
    await asyncio.gather(*tasks)

async def main():
    print(f"\n🔥 INICIANDO CRAWLER AGRESIVO SOBRE: {URL_INICIAL} 🔥")
    async with httpx.AsyncClient(http2=True, follow_redirects=True) as client:
        await crawl(client, URL_INICIAL)
    print(f"\n📦 Total URLs rastreadas: {len(visited)}")

if __name__ == "__main__":
    asyncio.run(main())
