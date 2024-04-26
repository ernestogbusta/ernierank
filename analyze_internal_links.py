#analyze_internal_links.py

import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Optional, Any, Tuple
import asyncio
from pydantic import BaseModel, HttpUrl, ValidationError, Field
from fastapi import Body
import logging
from httpx import AsyncClient, Timeout
import xmltodict
import re

class LinkAnalysis(BaseModel):
    source_url: HttpUrl = Field(..., description="URL of the page where the link is located")
    target_url: HttpUrl = Field(..., alias='url', description="URL the link points to")
    anchor_text: str
    seo_quality: str = "Unknown"
    similarity_score: float = 0.0

class InternalLinkAnalysis(BaseModel):
    domain: HttpUrl
    internal_links_data: List[LinkAnalysis]

async def get_http_client():
    return httpx.AsyncClient(
        timeout=Timeout(30.0, connect=5.0, read=60.0, write=60.0),
        follow_redirects=True
    )

def correct_url_format(url: str) -> str:
    if not url.startswith(('http://', 'https://')):
        return f'https://{url}'
    return url

async def analyze_internal_links(domain: str, client: httpx.AsyncClient) -> InternalLinkAnalysis:
    corrected_domain = correct_url_format(domain)
    try:
        response = await client.get(corrected_domain)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            internal_links_data = []
            for link in soup.find_all('a', href=True):
                link_url = urljoin(corrected_domain, link['href'])
                if urlparse(link_url).netloc == urlparse(corrected_domain).netloc:
                    seo_quality, similarity_score = await evaluate_link_quality_and_similarity(link.get_text(strip=True), link_url, client)
                    internal_links_data.append(LinkAnalysis(
                        source_url=corrected_domain,
                        url=link_url,
                        anchor_text=link.get_text(strip=True),
                        seo_quality=seo_quality,
                        similarity_score=similarity_score
                    ))
            return InternalLinkAnalysis(domain=corrected_domain, internal_links_data=internal_links_data)
        else:
            logging.error(f"Failed to fetch {corrected_domain}: HTTP {response.status_code}")
            return InternalLinkAnalysis(domain=corrected_domain, internal_links_data=[])
    except Exception as e:
        logging.error(f"Error analyzing internal links for {domain}: {e}")
        return InternalLinkAnalysis(domain=corrected_domain, internal_links_data=[])

async def evaluate_link_quality_and_similarity(anchor_text: str, target_url: str, client: httpx.AsyncClient):
    try:
        response = await client.get(target_url, follow_redirects=True)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.text if soup.title else ""
            slug = urlparse(target_url).path.split('/')[-1].replace('-', ' ')
            meta_description = soup.find('meta', attrs={'name': 'description'})
            description = meta_description['content'] if meta_description else ""
            headers = ' '.join([h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])])

            # Calcular similaridad
            anchor_text_lower = anchor_text.lower()
            similarity_score = calculate_similarity(anchor_text_lower, title.lower() + " " + slug.lower())

            # Evaluar calidad SEO
            seo_quality = evaluate_seo_quality(anchor_text_lower, title, description, headers)

            return seo_quality, similarity_score
        else:
            logging.error(f"Failed to fetch {target_url}: HTTP {response.status_code}")
            return "Error", 0.0
    except Exception as e:
        logging.error(f"Error fetching URL {target_url}: {e}")
        return "Error", 0.0

def calculate_similarity(anchor_text: str, target_text: str):
    anchor_words = set(anchor_text.lower().split())
    target_words = set(target_text.lower().split())
    if not anchor_words:
        return 0.0
    common_words = anchor_words.intersection(target_words)
    return len(common_words) / len(anchor_words)

def evaluate_seo_quality(anchor_text, title, description, headers):
    text_elements = [title, description, headers]
    matches = sum(anchor_text in element.lower() for element in text_elements if element)
    if matches >= 2:
        return 'Muy bueno'
    elif matches == 1:
        return 'Bueno'
    return 'Necesita optimización'

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

def clean_url(url: str) -> str:
    """Clean and correct the URL format."""
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

