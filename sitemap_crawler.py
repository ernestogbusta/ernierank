import aiohttp
import xml.etree.ElementTree as ET

class SitemapExtractor:
    def __init__(self, sitemap_url):
        self.sitemap_url = sitemap_url

    async def fetch_sitemap(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    print(f"Invalid response {response.status} for {url}")
                    return None

    async def parse_sitemap(self, content):
        urls = []
        try:
            root = ET.fromstring(content)
            namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            for sitemap in root.findall('sitemap:sitemap', namespace):
                loc = sitemap.find('sitemap:loc', namespace)
                if loc is not None:
                    urls.append(loc.text)
            for url in root.findall('sitemap:url', namespace):
                loc = url.find('sitemap:loc', namespace)
                if loc is not None:
                    urls.append(loc.text)
        except ET.ParseError as e:
            print(f"Error parsing sitemap content: {e}")
        return urls

    async def crawl_sitemap(self):
        content = await self.fetch_sitemap(self.sitemap_url)
        if content is None:
            return []

        # Check if it's a sitemap index
        if "<sitemapindex" in content:
            urls = []
            sitemap_urls = await self.parse_sitemap(content)
            for sitemap_url in sitemap_urls:
                sitemap_content = await self.fetch_sitemap(sitemap_url)
                if sitemap_content:
                    urls.extend(await self.parse_sitemap(sitemap_content))
            return urls
        else:
            return await self.parse_sitemap(content)
