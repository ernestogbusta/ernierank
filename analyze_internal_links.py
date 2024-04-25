#analyze_internal_links.py

import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Optional
import asyncio
from pydantic import BaseModel, HttpUrl
from fastapi import Body

class LinkAnalysis(BaseModel):
    url: HttpUrl
    anchor_text: str
    seo_quality: Optional[str] = None
    similarity_score: Optional[float] = None

class InternalLinkAnalysis(BaseModel):
    domain: HttpUrl
    internal_links_data: Optional[List[LinkAnalysis]] = []

async def get_http_client():
    return httpx.AsyncClient(timeout=Timeout(30.0, connect=5.0, read=60.0, write=60.0))

async def analyze_internal_links(domain: str, client: httpx.AsyncClient) -> InternalLinkAnalysis:
    internal_links_data = generate_internal_links_data(domain)
    return InternalLinkAnalysis(domain=domain, internal_links_data=internal_links_data)

async def fetch_sitemap_for_internal_links(client: httpx.AsyncClient, url: str) -> List[str]:
    try:
        # Aumentar el timeout específico para esta solicitud
        response = await client.get(url, timeout=Timeout(30.0, read=60.0))
        if response.status_code == 200:
            sitemap_contents = xmltodict.parse(response.content)
            urls = []
            if 'sitemapindex' in sitemap_contents:
                for sitemap in sitemap_contents['sitemapindex']['sitemap']:
                    child_urls = await fetch_sitemap_for_internal_links(client, sitemap['loc'])
                    urls.extend(child_urls)
            elif 'urlset' in sitemap_contents:
                urls = [url_entry['loc'] for url_entry in sitemap_contents['urlset']['url']
                        if not url_entry['loc'].endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp'))]
            return urls
        else:
            logging.warning(f"Received non-200 status code {response.status_code} for {url}")
            return []
    except Exception as e:
        logging.error(f"Exception occurred while fetching sitemap from {url}: {e}")
        return []

async def process_internal_links(client: httpx.AsyncClient, urls: List[str], domain: str) -> List[LinkAnalysis]:
    internal_links_data = []
    for url in urls:
        page_content = await fetch_page_content(url, client)
        if page_content:
            soup = BeautifulSoup(page_content, 'html.parser')
            links = extract_internal_links(soup, url, domain)
            for link in links:
                # Asegúrate de usar await aquí para la función coroutine
                seo_quality, similarity_score = await evaluate_link_quality_and_similarity(link['anchor_text'], link['url'])
                internal_links_data.append(LinkAnalysis(url=link['url'], anchor_text=link['anchor_text'], seo_quality=seo_quality, similarity_score=similarity_score))
    return internal_links_data

async def evaluate_link_quality_and_similarity(anchor_text: str, target_url: str) -> (str, float):
    """
    Evalúa la calidad de SEO y calcula la similitud de contenido basado en el texto del ancla y el slug de la URL de destino.
    """
    content_keywords = extract_keywords_from_page(target_url)
    anchor_keywords = set(anchor_text.lower().split())
    content_keywords_set = set(content_keywords)
    overlap = anchor_keywords.intersection(content_keywords_set)
    similarity_score = len(overlap) / len(anchor_keywords) if anchor_keywords else 0

    seo_quality = 'Excellent' if similarity_score > 0.5 else 'Good' if similarity_score > 0 else 'Needs improvement'
    return seo_quality, similarity_score

async def fetch_page_content(url: str, client: httpx.AsyncClient) -> Optional[str]:
    try:
        # Aplicar timeout personalizado para esta operación
        response = await client.get(url, follow_redirects=True, timeout=Timeout(20.0))
        if response.status_code == 200:
            return response.text
        elif response.status_code in [301, 302]:  # Si hay redirecciones, manejarlas manualmente
            new_url = response.headers.get('Location')
            if new_url:
                return await fetch_page_content(new_url, client)
        else:
            logging.error(f"Failed to fetch {url}: HTTP {response.status_code}")
    except httpx.RequestError as e:
        logging.error(f"Request error while fetching {url}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error fetching {url}: {e}")
    return None

def extract_internal_links(soup: BeautifulSoup, base_url: str, domain: str) -> List[Dict[str, str]]:
    links = []
    domain_netloc = urlparse(domain).netloc
    for a_tag in soup.find_all('a', href=True):
        link_url = urljoin(base_url, a_tag['href'])
        link_netloc = urlparse(link_url).netloc
        # Verifica que el enlace sea interno y no apunte a secciones inútiles como anclas o javascript
        if link_netloc == domain_netloc and not urlparse(link_url).path.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
            links.append({
                'url': link_url,
                'anchor_text': a_tag.get_text(strip=True) or 'No Text'
            })
    print(f"Enlaces internos extraídos de {base_url}: {len(links)}")
    return links

async def analyze_url_for_internal_links(client: httpx.AsyncClient, url: str, domain: str) -> Optional[List[LinkAnalysis]]:
    try:
        response = await client.get(url, follow_redirects=True)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            links = extract_internal_links(soup, url, domain)
            if links:
                detailed_links = await evaluate_internal_links(links, url, client)
                return detailed_links
            else:
                print(f"No se encontraron enlaces internos en {url}.")
        else:
            print(f"Error al obtener {url}, código de estado HTTP: {response.status_code}")
    except httpx.RequestError as e:
        print(f"HTTP request failed for {url}: {e}")
    except Exception as e:
        print(f"Unexpected error occurred for {url}: {e}")
    return None

async def evaluate_internal_links(links: List[Dict[str, str]], source_url: str, client: httpx.AsyncClient) -> List[LinkAnalysis]:
    results = []
    source_soup = await fetch_soup(source_url, client)
    if not source_soup:
        return []

    for link in links:
        target_soup = await fetch_soup(link['url'], client)
        if not target_soup:
            results.append(LinkAnalysis(url=link['url'], anchor_text=link['anchor_text'], seo_quality='Target page could not be fetched', similarity_score=0))
        else:
            seo_quality, similarity_score = await evaluate_link_quality_and_similarity(link['anchor_text'], link['url'])
            results.append(LinkAnalysis(url=link['url'], anchor_text=link['anchor_text'], seo_quality=seo_quality, similarity_score=similarity_score))
    return results

async def fetch_soup(url: str, client: httpx.AsyncClient) -> Optional[BeautifulSoup]:
    try:
        response = await client.get(url, follow_redirects=True)
        if response.status_code == 200:
            print(f"Página recuperada correctamente para scraping: {url}")
            return BeautifulSoup(response.content, 'html.parser')
        else:
            logging.warning(f"Error HTTP {response.status_code} al intentar recuperar la URL: {url}")
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP status error while fetching URL {url}: {e.response.status_code}")
    except httpx.RequestError as e:
        logging.error(f"Request error while fetching URL {url}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching URL {url}: {e}")
    return None

def generate_internal_links_data(domain):
    # Esta es una función simulada que genera datos de ejemplo para los enlaces internos
    # Podrías adaptarla para que procese datos reales basados en el dominio
    sample_data = [
        {"url": f"{domain}/about", "anchor_text": "About Us", "seo_quality": "Good", "similarity_score": 0.9},
        {"url": f"{domain}/contact", "anchor_text": "Contact Us", "seo_quality": "Excellent", "similarity_score": 0.8}
    ]
    return [LinkAnalysis(**link) for link in sample_data]

def extract_keywords_from_page(url: str) -> List[str]:
    """
    Extrae palabras clave a partir del slug de la URL.
    """
    from urllib.parse import urlparse
    path = urlparse(url).path
    slug = path.strip('/').split('/')[-1]  # Toma el último segmento del path como slug
    keywords = slug.replace('-', ' ').split()
    return keywords

def is_internal_link(link: str, base_url: str) -> bool:
    parsed_link = urlparse(link)
    parsed_base = urlparse(base_url)
    
    # Asegura que el esquema de la URL es válido y pertenece al mismo dominio
    if parsed_link.scheme in ['http', 'https'] and parsed_link.netloc == parsed_base.netloc:
        return True
    
    # Verifica también subdominios como parte del mismo sitio web
    domain_link = '.'.join(parsed_link.netloc.split('.')[-2:])
    domain_base = '.'.join(parsed_base.netloc.split('.')[-2:])
    return domain_link == domain_base
