#analyze_internal_links.py

import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Optional, Any
import asyncio
from pydantic import BaseModel, HttpUrl, ValidationError
from fastapi import Body
import logging
from httpx import AsyncClient, Timeout
import xmltodict
import re

class LinkAnalysis(BaseModel):
    url: HttpUrl
    anchor_text: str
    seo_quality: Optional[str] = None
    similarity_score: Optional[float] = None
    context: Optional[str] = None
    relevance: Optional[str] = None

class InternalLinkAnalysis(BaseModel):
    domain: HttpUrl
    internal_links_data: Optional[List[LinkAnalysis]] = []

async def get_http_client():
    return httpx.AsyncClient(timeout=Timeout(30.0, connect=5.0, read=60.0, write=60.0))

async def analyze_internal_links(domain: str, client: httpx.AsyncClient) -> InternalLinkAnalysis:
    url = f"{domain}/sitemap.xml"  # Suponiendo que quieres analizar desde el sitemap
    try:
        urls = await fetch_sitemap_for_internal_links(client, url)
        if not urls:
            return InternalLinkAnalysis(domain=domain, internal_links_data=[])
        
        internal_links_data = await process_internal_links(client, urls, domain)
        return InternalLinkAnalysis(domain=domain, internal_links_data=internal_links_data)
    except Exception as e:
        logging.error(f"Error during internal link analysis for {domain}: {e}")
        return InternalLinkAnalysis(domain=domain, internal_links_data=[])


async def fetch_sitemap_for_internal_links(client: httpx.AsyncClient, url: str, max_depth=5, current_depth=0) -> List[str]:
    if current_depth > max_depth:
        logging.warning(f"Max recursion depth reached for sitemap: {url}")
        return []

    try:
        response = await client.get(url, follow_redirects=True)
        if response.status_code == 200:
            sitemap_contents = xmltodict.parse(response.content)
            urls = []

            if 'sitemapindex' in sitemap_contents:
                for sitemap in sitemap_contents['sitemapindex']['sitemap']:
                    child_url = urljoin(url, sitemap['loc'])  # Ensure absolute URLs
                    child_urls = await fetch_sitemap_for_internal_links(client, child_url, max_depth, current_depth + 1)
                    urls.extend(child_urls)
            elif 'urlset' in sitemap_contents:
                urls = [
                    urljoin(url, url_entry['loc'])  # Convert to absolute URLs
                    for url_entry in sitemap_contents['urlset']['url']
                    if not re.search(r'\.(jpg|jpeg|png|gif|svg|webp)$', url_entry['loc'], re.IGNORECASE)
                ]

            return urls
        else:
            logging.warning(f"Received non-200 status code {response.status_code} for {url}")
            return []

    except Exception as e:
        logging.error(f"Exception occurred while fetching sitemap from {url}: {e}")
        return []     

async def process_internal_links(client: httpx.AsyncClient, urls: List[str], domain: str) -> List[LinkAnalysis]:
    internal_links_data = []
    if not urls:
        logging.info(f"No URLs found for internal link analysis for {domain}.")
        return internal_links_data

    tasks = []
    for url in urls:
        task = asyncio.create_task(process_single_url(client, url, domain, internal_links_data))
        tasks.append(task)
    await asyncio.gather(*tasks)

    if not internal_links_data:
        logging.info(f"No internal links data gathered from URLs for {domain}.")

    return internal_links_data

async def process_single_url(client, url, domain, internal_links_data):
    page_content = await fetch_page_content(url, client)
    if page_content:
        soup = BeautifulSoup(page_content, 'html.parser')
        links = extract_internal_links(soup, url, domain)
        for link in links:
            cleaned_url = clean_url(link['url'])
            if cleaned_url:
                try:
                    seo_quality, similarity_score = await evaluate_link_quality_and_similarity(link['anchor_text'], cleaned_url)
                    context_and_relevance = analyze_link_context_and_relevance(soup, cleaned_url, domain)
                    internal_links_data.append(LinkAnalysis(
                        url=cleaned_url,
                        anchor_text=link['anchor_text'],
                        seo_quality=seo_quality,
                        similarity_score=similarity_score,
                        context=context_and_relevance['context'],
                        relevance=context_and_relevance['relevance']
                    ))
                except ValidationError as e:
                    logging.error(f"Validation error for URL {cleaned_url}: {e}")
                except Exception as e:
                    logging.error(f"Error processing URL {cleaned_url}: {e}")
    else:
        logging.error(f"Failed to fetch or parse content for URL {url}, which might be affecting internal links extraction.")


async def fetch_http_status(url: str, client: httpx.AsyncClient) -> int:
    """ Fetch the HTTP status for a URL to determine its accessibility """
    try:
        response = await client.head(url)
        return response.status_code
    except Exception as e:
        logging.error(f"Failed to fetch HTTP status for {url}: {e}")
        return 0  # Indicating an unreachable URL


async def evaluate_link_quality_and_similarity(anchor_text: str, target_url: str) -> (str, float):
    """
    Evalúa la calidad de SEO y calcula la similitud de contenido basado en el texto del ancla y el slug de la URL de destino.
    """
    content_keywords = extract_keywords_from_page(target_url)
    expanded_anchor_text = expand_with_synonyms(anchor_text)  # Ampliar texto del ancla con sinónimos
    anchor_keywords = set(expanded_anchor_text.lower().split())
    content_keywords_set = set(content_keywords)
    overlap = anchor_keywords.intersection(content_keywords_set)
    similarity_score = len(overlap) / len(anchor_keywords) if anchor_keywords else 0

    seo_quality = 'Excellent' if similarity_score > 0.5 else 'Good' if similarity_score > 0 else 'Needs improvement'
    return seo_quality, similarity_score

async def fetch_page_content(url: str, client: httpx.AsyncClient) -> Optional[str]:
    try:
        response = await client.get(url, follow_redirects=True, timeout=Timeout(20.0))
        if response.status_code == 200:
            return response.text
        elif response.status_code in [301, 302]:
            new_url = response.headers.get('Location')
            if new_url:
                # Asegurar que la nueva URL es absoluta
                new_url = urljoin(url, new_url)
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
        # Solo considera enlaces internos y asegura que no sean enlaces a secciones inútiles
        if link_netloc == domain_netloc and not urlparse(link_url).path.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
            links.append({
                'url': link_url,
                'anchor_text': a_tag.get_text(strip=True) or 'No Text'
            })
    return links

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

def analyze_link_context_and_relevance(soup: BeautifulSoup, link_url: str, domain: str) -> Dict[str, str]:
    # Extraer el contexto del contenido alrededor del enlace
    context = ''
    for a_tag in soup.find_all('a', href=True):
        if a_tag['href'] == link_url:
            context = a_tag.find_parent('p').text if a_tag.find_parent('p') else ''
            break
    
    # Simulación de análisis de relevancia
    relevance = 'High' if 'example keyword' in context else 'Low'
    
    return {'context': context, 'relevance': relevance}

def get_surrounding_text(a_tag):
    # Extrae el texto circundante para entender mejor el contexto del enlace
    parent = a_tag.parent
    previous_sibling = a_tag.find_previous_sibling()
    next_sibling = a_tag.find_next_sibling()
    surrounding_text = ''
    if previous_sibling:
        surrounding_text += previous_sibling.get_text() + ' '
    surrounding_text += a_tag.get_text() + ' '
    if next_sibling:
        surrounding_text += next_sibling.get_text()
    return surrounding_text

def calculate_content_relevance(surrounding_text, page_content):
    # Un ejemplo simple de cálculo de relevancia, debería mejorar para incluir análisis semántico
    if surrounding_text in page_content:
        return 'High'
    return 'Low'

def expand_with_synonyms(text):
    synonyms = {
        'buy': ['purchase', 'acquire', 'obtain'],
        'learn': ['study', 'understand', 'grasp'],
        # Añade más sinónimos según necesidad
    }
    words = text.split()
    expanded_text = []
    for word in words:
        if word.lower() in synonyms:
            expanded_text.extend(synonyms[word.lower()])
        else:
            expanded_text.append(word)
    return ' '.join(expanded_text)

def generate_executive_summary(links_data: List[LinkAnalysis]) -> Dict[str, Any]:
    total_links = len(links_data)
    high_quality_count = sum(1 for link in links_data if link.seo_quality == 'Excellent')
    high_relevance_count = sum(1 for link in links_data if getattr(link, 'relevance', '') == 'High')

    summary = {
        'total_links': total_links,
        'high_quality_links_percentage': (high_quality_count / total_links * 100) if total_links else 0,
        'high_relevance_links_percentage': (high_relevance_count / total_links * 100) if total_links else 0,
        'suggestions': 'Consider revising low relevance links and improving anchor text diversity.'
    }
    return summary

def correct_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme:
        return f"http://{url}"  # Asume http por defecto si no hay esquema
    return url

def clean_url(url: str) -> str:
    # Corrige problemas comunes en las URLs como espacios no deseados y esquemas incorrectos.
    url = url.strip()
    url = url.replace(" ", "")
    if "://" not in url:
        return "https://" + url
    parts = url.split("://")
    corrected_scheme = parts[0].lower()
    rest = parts[1]
    if not rest.startswith("//"):
        return corrected_scheme + "://" + rest
    return url

