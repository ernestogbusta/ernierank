# Este es el archivo main.py, dame código robusto y completo para solucionar el problema pero dame ÚNICAMENTE las funciones o endpoints que debo modificar o actualizar, EN NINGUN CASO me des funciones o endpoints que ya estén funcionando bien en mi código

from analyze_url import analyze_url
from analyze_internal_links import analyze_internal_links, InternalLinkAnalysis, correct_url_format
from analyze_wpo import analyze_wpo
from analyze_cannibalization import analyze_cannibalization
from analyze_thin_content import analyze_thin_content, ThinContentRequest
from generate_content import generate_seo_content, process_new_data
from analyze_404 import fetch_urls, check_url, crawl_site, find_broken_links
from analyze_robots import fetch_robots_txt, analyze_robots_txt, RobotsTxtRequest
from fastapi import FastAPI, HTTPException, Depends, Body, Request, BackgroundTasks, Response
import httpx
from httpx import AsyncClient, Timeout, RemoteProtocolError
from bs4 import BeautifulSoup
import xmltodict
import os
import json
from pydantic import BaseModel, HttpUrl, validator
import uvicorn
from collections import Counter
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse, urljoin
import urllib.parse
import re
import asyncio
import time
import requests
from starlette.middleware.gzip import GZipMiddleware
import pytrends
from pytrends.request import TrendReq
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import sys
import random
import gzip
from requests.exceptions import HTTPError, RequestException
import aiohttp

app = FastAPI(title="ErnieRank API")

@app.on_event("startup")
async def startup_event():
    app.state.openai_api_key = os.getenv("OPENAI_API_KEY")
    if not app.state.openai_api_key:
        print("Failed to detect OPENAI_API_KEY:", file=sys.stderr)
        raise RuntimeError("OPENAI_API_KEY is not set in the environment variables")
    else:
        print("OPENAI_API_KEY detected successfully.", file=sys.stderr)

    timeout = httpx.Timeout(30.0, connect=10.0, read=20.0, write=10.0, pool=20.0)
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }

    app.state.client = httpx.AsyncClient(
        http1=True,
        timeout=timeout,
        limits=limits,
        headers=headers,  # 👉 Ahora SIEMPRE va como navegador
        follow_redirects=True
    )

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.client.aclose()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.head("/")
def read_root():
    return Response(content=None, status_code=200)

########## ANALYZE_URL ############

class BatchRequest(BaseModel):
    domain: str
    batch_size: int = 100  # valor por defecto
    start: int = 0        # valor por defecto para iniciar, asegura que siempre tenga un valor


@app.post("/process_urls_in_batches")
async def process_urls_in_batches(request: BatchRequest):
    domain = request.domain.rstrip('/')
    print(f"Fetching URLs from domain: {domain}")
    
    urls = await fetch_sitemap(app.state.client, domain)
    if not urls:
        print("🚫 No URLs found in sitemap.")
        return {"processed_urls": [], "more_batches": False, "next_batch_start": 0}
    
    print(f"✅ Total URLs fetched: {len(urls)}")
    urls_to_process = urls[request.start:request.start + request.batch_size]
    if not urls_to_process:
        print("🚫 No URLs to process in this batch.")
        return {"processed_urls": [], "more_batches": False, "next_batch_start": 0}

    semaphore = asyncio.Semaphore(5)

    async def safe_analyze(url):
        async with semaphore:
            return await retry_analyze_url(url, app.state.client)

    tasks = [safe_analyze(url) for url in urls_to_process]
    results = await asyncio.gather(*tasks)
    
    valid_results = [
        {
            "url": result['url'],
            "title": result.get('title', "No title provided"),
            "meta_description": result.get('meta_description', "No description provided"),
            "slug": urlparse(result['url']).path,
            "h1_tags": result.get('h1_tags', []),
            "h2_tags": result.get('h2_tags', []),
            "main_keyword": result.get('main_keyword', "Not specified"),
            "secondary_keywords": result.get('secondary_keywords', []),
            "semantic_search_intent": result.get('semantic_search_intent', "Not specified")
        }
        for result in results if result
    ]

    next_start = request.start + len(urls_to_process)
    more_batches = next_start < len(urls)

    return {
        "processed_urls": valid_results,
        "more_batches": more_batches,
        "next_batch_start": next_start if more_batches else 0
    }

    async def sem_analyze_url(url):
        async with semaphore:
            try:
                result = await retry_analyze_url(url, app.state.client)
                await asyncio.sleep(2)  # 💤 Espera de 2 segundos entre peticiones para evitar baneos
                return result
            except Exception as e:
                print(f"❌ Error procesando {url}: {e}")
                return None

    tasks = [sem_analyze_url(url) for url in urls_to_process]
    results = await asyncio.gather(*tasks)

    print(f"✅ Results received for batch: {len([r for r in results if r])} successful, {len(results) - len([r for r in results if r])} failed.")

    valid_results = [
        {
            "url": result['url'],
            "title": result.get('title', "No title provided"),
            "meta_description": result.get('meta_description', "No description provided"),
            "slug": urlparse(result['url']).path,
            "h1_tags": result.get('h1_tags', []),
            "h2_tags": result.get('h2_tags', []),
            "main_keyword": result.get('main_keyword', "Not specified"),
            "secondary_keywords": result.get('secondary_keywords', []),
            "semantic_search_intent": result.get('semantic_search_intent', "Not specified")
        }
        for result in results if result
    ]

    next_start = request.start + len(urls_to_process)
    more_batches = next_start < len(urls)

    print(f"🔄 More batches pending: {more_batches} | Next batch start index: {next_start}")

    return {
        "processed_urls": valid_results,
        "more_batches": more_batches,
        "next_batch_start": next_start if more_batches else 0
    }


    async def sem_analyze_url(url):
        async with semaphore:
            try:
                result = await retry_analyze_url(url, app.state.client)
                await asyncio.sleep(2)  # 💤 Añadir espera para evitar bloqueos
                return result
            except Exception as e:
                print(f"❌ Error procesando {url}: {e}")
                return None

    tasks = [sem_analyze_url(url) for url in urls_to_process]
    results = await asyncio.gather(*tasks)

    print(f"✅ Results received for batch: {len([r for r in results if r])} successful, {len(results) - len([r for r in results if r])} failed.")

    valid_results = [
        {
            "url": result['url'],
            "title": result.get('title', "No title provided"),
            "meta_description": result.get('meta_description', "No description provided"),
            "slug": urlparse(result['url']).path,
            "h1_tags": result.get('h1_tags', []),
            "h2_tags": result.get('h2_tags', []),
            "main_keyword": result.get('main_keyword', "Not specified"),
            "secondary_keywords": result.get('secondary_keywords', []),
            "semantic_search_intent": result.get('semantic_search_intent', "Not specified")
        }
        for result in results if result
    ]

    next_start = request.start + len(urls_to_process)
    more_batches = next_start < len(urls)

    print(f"🔄 More batches pending: {more_batches} | Next batch start index: {next_start}")

    return {
        "processed_urls": valid_results,
        "more_batches": more_batches,
        "next_batch_start": next_start if more_batches else 0
    }



async def sem_analyze_url(url):
    async with semaphore:
        try:
            result = await retry_analyze_url(url, app.state.client)
            await asyncio.sleep(2)  # 💤 Añadimos 2 segundos de espera entre peticiones
            return result
        except Exception as e:
            print(f"❌ Error procesando {url}: {e}")
            return None

    print(f"✅ Results received for batch: {len([r for r in results if r])} successful, {len(results) - len([r for r in results if r])} failed.")

    valid_results = [
        {
            "url": result['url'],
            "title": result.get('title', "No title provided"),
            "meta_description": result.get('meta_description', "No description provided"),
            "main_keyword": result.get('main_keyword', "Not specified"),
            "secondary_keywords": result.get('secondary_keywords', []),
            "semantic_search_intent": result.get('semantic_search_intent', "Not specified"),
            "content": result.get('content', "")
        }
        for result in results if result
    ]
    print(f"✅ Filtered valid results: {len(valid_results)}")

    next_start = request.start + len(urls_to_process)
    more_batches = next_start < len(urls)
    print(f"🔄 More batches pending: {more_batches} | Next batch start index: {next_start}")

    return {
        "processed_urls": valid_results,
        "more_batches": more_batches,
        "next_batch_start": next_start if more_batches else None
    }

async def sem_analyze_url(url):
    async with semaphore:
        return await retry_analyze_url(url, app.state.client)

    tasks = [sem_analyze_url(url) for url in urls_to_process]
    results = await asyncio.gather(*tasks)
    print(f"Results received: {results}")

    valid_results = [
        {
            "url": result['url'],
            "title": result.get('title', "No title provided"),
            "meta_description": result.get('meta_description', "No description provided"),
            "main_keyword": result.get('main_keyword', "Not specified"),
            "secondary_keywords": result.get('secondary_keywords', []),
            "semantic_search_intent": result.get('semantic_search_intent', "Not specified"),
            "content": result.get('content', "")
        }
        for result in results if result
    ]
    print(f"Filtered results: {valid_results}")

    next_start = request.start + len(urls_to_process)
    more_batches = next_start < len(urls)
    print(f"More batches: {more_batches}, Next batch start index: {next_start}")

    return {
        "processed_urls": valid_results,
        "more_batches": more_batches,
        "next_batch_start": next_start if more_batches else None
    }

async def try_fetch_and_parse_sitemap(client: httpx.AsyncClient, sitemap_url: str, headers: dict, retries: int = 3) -> list:
    for attempt in range(retries):
        try:
            response = await client.get(sitemap_url, headers=headers, timeout=30, follow_redirects=True)
            if response.status_code == 200:
                return await parse_sitemap(response, sitemap_url, client, headers)
        except (httpx.RequestError, httpx.RemoteProtocolError) as e:
            print(f"⚠️ Error de conexión en {sitemap_url}: {e} (Intento {attempt+1}/{retries})")
            await asyncio.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"❌ Error inesperado en {sitemap_url}: {e}")
            break
    return []

async def find_sitemaps_in_html(client: httpx.AsyncClient, base_domain: str, headers: dict):
    try:
        response = await client.get(base_domain, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            sitemap_urls = [
                urljoin(base_domain, a['href'])
                for a in soup.find_all('a', href=True)
                if 'sitemap' in a['href'] and a['href'].endswith(('.xml', '.gz'))
            ]
            urls_collected = set()
            for sitemap_url in sitemap_urls:
                urls = await try_fetch_and_parse_sitemap(client, sitemap_url, headers)  # ⚡ Aquí estaba mal: ahora correcto
                if urls:
                    urls_collected.update(urls)
            return urls_collected
    except Exception as e:
        print(f"⚠️ Error buscando sitemaps en HTML {base_domain}: {e}")
    return set()

async def fetch_with_retry(client, url, headers, retries=3, delay=5):
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"🌎 Intento {attempt}: Fetching {url}")
            response = await client.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            logger.info(f"✅ Success: {url}")
            return response
        except (httpx.RequestError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            logger.warning(f"⚠️ Error {e} in {url}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
        except Exception as e:
            logger.error(f"❌ Fatal error fetching {url}: {e}")
            break
    logger.error(f"🛑 Failed after {retries} attempts: {url}")
    return None

async def fetch_sitemap(client: httpx.AsyncClient, base_url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "application/xml,text/xml,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }

    base_domain = urlparse(base_url).scheme + "://" + urlparse(base_url).netloc
    sitemap_candidates = [
        f"{base_domain}/sitemap_index.xml",
        f"{base_domain}/sitemap.xml",
        f"{base_domain}/sitemap1.xml",
        f"{base_domain}/sitemap.xml.gz"
    ]

    collected_urls = set()

    # 1. Probar los candidatos conocidos
    for sitemap_url in sitemap_candidates:
        urls = await try_fetch_and_parse_sitemap(client, sitemap_url, headers)
        if urls:
            collected_urls.update(urls)
            break  # ✅ Si uno funciona, no sigas probando los demás

    # 2. Buscar en robots.txt si no encontró antes
    if not collected_urls:
        robots_sitemaps = await discover_sitemaps_from_robots_txt(client, base_domain, headers)
        for sitemap_url in robots_sitemaps:
            urls = await try_fetch_and_parse_sitemap(client, sitemap_url, headers)
            if urls:
                collected_urls.update(urls)

    # 3. Buscar en la home HTML
    if not collected_urls:
        urls = await find_sitemaps_in_html(client, base_domain, headers)
        collected_urls.update(urls)

    if not collected_urls:
        print(f"🚫 No se encontraron URLs en {base_domain}")
        return None

    print(f"✅ Total URLs encontradas: {len(collected_urls)}")
    return list(collected_urls)

async def parse_sitemap(response: httpx.Response, sitemap_url: str, client: httpx.AsyncClient, headers: dict):
    try:
        content = gzip.decompress(response.content) if sitemap_url.endswith('.gz') else response.content
        # Primera verificación rápida
        if not content.lstrip().startswith(b"<"):
            print(f"⚠️ No parece XML válido en {sitemap_url}")
            return []

        data = xmltodict.parse(content)

        if 'urlset' in data:
            urls = data['urlset'].get('url', [])
            if isinstance(urls, dict):
                urls = [urls]
            return [entry['loc'] for entry in urls if 'loc' in entry]

        elif 'sitemapindex' in data:
            nested = data['sitemapindex'].get('sitemap', [])
            if isinstance(nested, dict):
                nested = [nested]
            all_nested_urls = []
            for sitemap in nested:
                loc = sitemap.get('loc')
                if loc:
                    nested_urls = await fetch_individual_sitemap(client, loc, headers)
                    if nested_urls:
                        all_nested_urls.extend(nested_urls)
            return all_nested_urls

    except Exception as e:
        print(f"⚠️ Error parseando sitemap {sitemap_url}: {e}")
        return []

async def fetch_individual_sitemap(client: httpx.AsyncClient, sitemap_url: str, headers: dict) -> list:
    try:
        response = await client.get(sitemap_url, headers=headers, timeout=30)
        if response.status_code == 200:
            content = gzip.decompress(response.content) if sitemap_url.endswith('.gz') else response.content
            data = xmltodict.parse(content)

            if 'urlset' in data:
                urls = data['urlset'].get('url', [])
                if isinstance(urls, dict):
                    urls = [urls]
                return [entry['loc'] for entry in urls if 'loc' in entry]

            elif 'sitemapindex' in data:
                nested = data['sitemapindex'].get('sitemap', [])
                if isinstance(nested, dict):
                    nested = [nested]
                all_nested_urls = []
                for sitemap in nested:
                    loc = sitemap.get('loc')
                    if loc:
                        nested_urls = await fetch_individual_sitemap(client, loc, headers)
                        if nested_urls:
                            all_nested_urls.extend(nested_urls)
                return all_nested_urls
    except Exception as e:
        print(f"⚠️ Error descargando o parseando sitemap individual {sitemap_url}: {e}")

    return []

async def discover_sitemaps_from_robots_txt(client: httpx.AsyncClient, base_domain: str, headers: dict) -> list:
    robots_url = f"{base_domain}/robots.txt"
    discovered = []

    try:
        response = await client.get(robots_url, headers=headers, timeout=10)
        if response.status_code == 200:
            lines = response.text.splitlines()
            for line in lines:
                if line.lower().startswith('sitemap:'):
                    sitemap_url = line.split(':', 1)[1].strip()
                    discovered.append(sitemap_url)
        else:
            print(f"⚠️ No se pudo acceder a robots.txt en {robots_url}")
    except Exception as e:
        print(f"⚠️ Error leyendo robots.txt en {robots_url}: {e}")

    return discovered

async def retry_analyze_url(url: str, client: httpx.AsyncClient, max_retries: int = 5, initial_delay: float = 1.5):
    """
    Reintenta analizar una URL varias veces con backoff exponencial si falla, usando headers de navegador.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            print(f"🔄 Attempt {attempt} fetching: {url}")
            result = await analyze_url(url, client)
            if result:
                return result
            else:
                print(f"⚠️ Empty result for {url} attempt {attempt}")
        except (httpx.RequestError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            print(f"⚠️ Connection error on {url} at attempt {attempt}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error on {url}: {e}")
            return None
        
        await asyncio.sleep(delay)
        delay *= 2  # Exponential backoff

    print(f"🛑 Failed fetching {url} after {max_retries} retries.")
    return None


############################################



########## ANALYZE_INTERNAL_LINKS ##########

@app.post("/analyze_internal_links", response_model=InternalLinkAnalysis)
async def handle_analyze_internal_links(domain: str = Body(..., embed=True)):
    corrected_domain = correct_url_format(domain)
    async with httpx.AsyncClient() as client:
        result = await analyze_internal_links(corrected_domain, client)
        return result


############################################




########## ANALYZE_WPO ##########


def check_server_availability(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Server not available for {url}: {str(e)}")
        return False

# Llamada a la función
if check_server_availability('https://example.com'):
    print("Procede con la obtención del tamaño de los recursos")
else:
    print("El servidor no está disponible. Intenta más tarde.")

class WPORequest(BaseModel):
    url: str

@app.post("/analyze_wpo")
async def analyze_wpo_endpoint(request: WPORequest):
    # Asegúrate de pasar la URL como argumento a la función analyze_wpo
    return await analyze_wpo(request.url)

###############################################



########### ANALIZE_CANNIBALIZATION ##########


class CannibalizationData(BaseModel):
    url: HttpUrl
    semantic_search_intent: str

class CannibalizationRequest(BaseModel):
    processed_urls: List[CannibalizationData]

@app.post("/analyze_cannibalization")
async def endpoint_analyze_cannibalization(request: CannibalizationRequest):
    try:
        results = await analyze_cannibalization(request.processed_urls)
        return {"message": "Análisis de canibalización completado correctamente", "cannibalization_issues": results}
    except HTTPException as e:
        return {"message": e.detail}


##############################################



########### GENERATE_CONTENT ##########


class ContentRequest(BaseModel):
    url: HttpUrl

@app.post("/generate_content")
async def generate_content_endpoint(request: Request):
    req_data = await request.json()
    url = req_data.get("url")

    if not url:
        raise HTTPException(status_code=422, detail="URL parameter is required.")

    try:
        new_data = await process_new_data(url, app.state.client)
        if not new_data:
            raise HTTPException(status_code=500, detail="Failed to process new data")

        headers = {
            "Authorization": f"Bearer {app.state.openai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {"role": "system", "content": "Please generate SEO content based on the following details."},
                {"role": "user", "content": f"Title: {new_data['title']}, Description: {new_data['meta_description']}, Main Keyword: {new_data['main_keyword']}"}
            ],
            "max_tokens": 1500,
            "temperature": 0.5
        }

        # Utiliza el cliente HTTP persistente para realizar la petición
        response = await app.state.client.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=httpx.Timeout(120.0, connect=120.0)
        )
        response.raise_for_status()  # Asegúrate de que no hay errores en la respuesta.
        content_generated = response.json()["choices"][0]["message"]["content"]
        return {"generated_content": content_generated}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=f"HTTP error: {exc.response.text}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=500, detail=f"HTTP request failed: {str(exc)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


##############################################


########### ANALYZE_THIN_CONTENT ##########

# Configurando el logger

class PageData(BaseModel):
    url: HttpUrl
    title: str
    meta_description: Optional[str] = None
    h1: Optional[str] = None
    h2: Optional[List[str]] = []
    main_keyword: Optional[str] = None
    secondary_keywords: List[str]
    semantic_search_intent: str

    @validator('h1', 'meta_description', 'main_keyword', pre=True, always=True)
    def ensure_not_empty(cls, v):
        if v == "":
            return None
        return v

    @validator('h2', pre=True, always=True)
    def ensure_list(cls, v):
        if v is None:
            return []
        return v

class ThinContentRequest(BaseModel):
    processed_urls: List[Dict[str, Any]]

@app.post("/analyze_thin_content")
async def analyze_thin_content_endpoint(request: Request):
    try:
        request_data = await request.json()
        try:
            thin_request = ThinContentRequest(**request_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing request data: {e}")

        if not thin_request.processed_urls:
            raise HTTPException(status_code=400, detail="No URLs provided")

        analysis_results = analyze_thin_content(thin_request.processed_urls)

        formatted_response = {
            "thin_content_pages": [
                {
                    "url": urllib.parse.urlparse(page["url"]).path,
                    "level": page["level"],
                    "description": page["details"]
                }
                for page in analysis_results["thin_content_urls"]
            ]
        }
        return formatted_response
    except HTTPException as http_exc:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/full_thin_content_analysis")
async def full_thin_content_analysis(request: Request):
    try:
        request_data = await request.json()
        
        batch_request = BatchRequest(**request_data)
        process_urls_response = await process_urls_in_batches(batch_request)

        if not process_urls_response:
            raise HTTPException(status_code=500, detail="Error processing URLs in batches")

        thin_content_request = ThinContentRequest(
            processed_urls=process_urls_response["processed_urls"]
        )

        analysis_results = analyze_thin_content(thin_content_request.processed_urls)

        formatted_response = {
            "thin_content_pages": [
                {
                    "url": urllib.parse.urlparse(page["url"]).path,
                    "level": page["level"],
                    "description": page["details"]
                }
                for page in analysis_results["thin_content_urls"]
            ]
        }
        return formatted_response

    except HTTPException as http_exc:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


#######################################



########### ANALYZE_404 ##########

class DomainRequest(BaseModel):
    domain: HttpUrl

async def fetch_page(url: str, client: httpx.AsyncClient):
    try:
        response = await client.get(url)
        if response.status_code == 404:
            return None, 404
        response.raise_for_status()
        return response.text, response.status_code
    except httpx.HTTPStatusError as e:
        return None, e.response.status_code
    except httpx.HTTPError:
        return None, 500

async def crawl_page(url: str, base_url: str, client: httpx.AsyncClient, visited: set):
    if url in visited:
        return []
    visited.add(url)
    content, status = await fetch_page(url, client)
    results = [{"url": url, "status": status}]
    if content is None:
        return results

    soup = BeautifulSoup(content, 'html.parser')
    links = [link.get('href') for link in soup.find_all('a', href=True)]
    internal_links = {urljoin(base_url, link) for link in links if link and (link.startswith('/') or base_url in link)}

    tasks = [crawl_page(link, base_url, client, visited) for link in internal_links]
    crawled_pages = await asyncio.gather(*tasks)
    for page in crawled_pages:
        results.extend(page)
    return results

@app.post("/check-domain/")
async def check_domain(request: DomainRequest):
    base_url = request.domain.rstrip('/')
    visited = set()
    async with httpx.AsyncClient(timeout=10.0) as client:
        results = await crawl_page(base_url, base_url, client, visited)
        return results

###################################


############ SEARCH_KEYWORDS #############


class KeywordRequest(BaseModel):
    topic: str

# Datos ejemplo - valores de volumen de búsqueda de Google Trends y Semrush
trends_volumes = np.array([14800, 12580, 11248, 9472, 9176, 8880, 7696, 6512, 6364, 6216, 5772, 4884, 4144, 3848, 3700, 3552, 3404, 2812, 2812, 2664, 2516, 2368, 2368]).reshape(-1, 1)
semrush_volumes = np.array([3600, 320, 4400, 320, 480, 590, 50, 390, 590, 390, 320, 260, 1600, 210, 170, 880, 720, 110, 90, 210, 170, 140, 170])

# Inicializar y ajustar un modelo de regresión polinomial
poly = PolynomialFeatures(degree=2)
trends_poly = poly.fit_transform(trends_volumes)
model = LinearRegression()
model.fit(trends_poly, semrush_volumes)

def adjust_volume(trends_volume):
    """ Ajusta el volumen basado en el modelo de regresión polinomial """
    trends_volume_transformed = poly.transform(np.array([[trends_volume]]))
    return model.predict(trends_volume_transformed)[0]

# Configuración inicial de pytrends
pytrends = TrendReq(hl='es-ES', tz=360)

async def fetch_google_search_results(topic, max_attempts=10):
    attempts = 0
    while attempts < max_attempts:
        try:
            pytrends.build_payload([topic])
            related_queries = pytrends.related_queries()
            return process_related_queries(related_queries, topic)
        
        except HTTPError as http_err:
            if http_err.response.status_code == 429:
                attempts += 1
                sleep_time = (2 ** attempts) * 5
                await asyncio.sleep(sleep_time * random.uniform(1.5, 2))
            else:
                raise HTTPException(status_code=http_err.response.status_code, detail="HTTP error occurred.")
        
        except RequestException as e:  # Captura de errores relacionados con peticiones HTTP
            attempts += 1
            sleep_time = (2 ** attempts) * 5
            await asyncio.sleep(sleep_time * random.uniform(1.5, 2))
            
            if attempts == max_attempts:
                raise HTTPException(status_code=500, detail=f"Max retries exceeded. Google Trends error: {str(e)}")
    
    raise HTTPException(status_code=500, detail="Could not retrieve Google Trends data after max attempts.")


def get_cached_keywords(topic):
    # Función de ejemplo que devuelve keywords cacheadas
    # Deberías implementar tu propia lógica para obtener keywords predefinidas o desde un caché
    cached_keywords = {
        "marketing": [{"keyword": "marketing digital", "volume": 12000}, {"keyword": "estrategias de marketing", "volume": 9500}],
        "tecnología": [{"keyword": "inteligencia artificial", "volume": 13000}, {"keyword": "nuevas tecnologías", "volume": 9000}],
    }
    return cached_keywords.get(topic, [])

@app.post("/search_keywords")
async def search_keywords(request: KeywordRequest):
    try:
        # Intentar obtener resultados desde Google Trends
        google_results = await fetch_google_search_results(request.topic)
    except HTTPException as e:
        # Si Google Trends falla, usa datos cacheados como fallback
        google_results = get_cached_keywords(request.topic)
        if not google_results:
            raise HTTPException(status_code=404, detail="No keywords found in Google Trends or cache.")
    
    if not google_results:
        raise HTTPException(status_code=404, detail="No keywords found.")
    
    return {"keywords": google_results}


scaling_factor = 14800 / 100  # Suponiendo que 100 es el máximo de Trends para 'Digital Marketing'

def process_related_queries(related_queries, topic):
    keywords = []
    excluded_words = ['la', 'el', 'los', 'las']

    if not related_queries or topic not in related_queries:
        raise HTTPException(status_code=404, detail="No related queries found for the topic.")
    
    related_data = related_queries.get(topic, {})
    if not related_data:  # Verificar si los datos relacionados están presentes
        raise HTTPException(status_code=404, detail="No related data found for the topic.")
    
    for data_type in ['top', 'rising']:
        if related_data.get(data_type) is not None:
            for query in related_data[data_type].to_dict('records'):
                if not any(excluded_word in query['query'].split() for excluded_word in excluded_words):
                    trends_volume = query['value'] * scaling_factor
                    if trends_volume > 0:
                        scaled_volume = adjust_volume(trends_volume)
                    else:
                        scaled_volume = 0
                    keywords.append({"keyword": query['query'], "volume": scaled_volume})

    return keywords


##########################################


############ ANALYZE_ROBOTS ##############


@app.post("/analyze-robots")
async def analyze_robots_endpoint(request: RobotsTxtRequest):
    """Endpoint para obtener y analizar el archivo robots.txt de un dominio dado."""
    robots_txt_content = await fetch_robots_txt(request.url)
    analysis_results = analyze_robots_txt(robots_txt_content)
    if not analysis_results:
        raise HTTPException(status_code=404, detail="No actionable rules found in robots.txt")
    return {"domain": request.url, "analysis": analysis_results}


##########################################



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="debug")