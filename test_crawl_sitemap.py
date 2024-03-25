# test_crawl_sitemap.py
from sitemap_crawler import crawl_sitemap

# Usa un dominio de ejemplo. Cambia esto por el dominio que desees probar.
domain_url = 'https://aulacm.com'

urls = crawl_sitemap(domain_url)
print(f"URLs encontradas: {len(urls)}")
for url in urls:
    print(url)
