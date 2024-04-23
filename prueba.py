import asyncio
import httpx

# Asumiendo que la función fetch_sitemap_for_internal_links ya está definida como se mostró anteriormente

async def main():
    async with httpx.AsyncClient() as client:
        urls = await fetch_sitemap_for_internal_links(client, 'https://aulacm.com/sitemap_index.xml')
        print(urls)

if __name__ == "__main__":
    asyncio.run(main())
