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
# 🧠 Memoria temporal para almacenar el progreso de batches
batch_results = {}

@app.on_event("startup")
async def startup_event():
    try:
        app.state.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not app.state.openai_api_key:
            print("Failed to detect OPENAI_API_KEY:", file=sys.stderr)
            raise RuntimeError("OPENAI_API_KEY is not set in the environment variables")
        else:
            print("OPENAI_API_KEY detected successfully.", file=sys.stderr)

        timeout = httpx.Timeout(60.0, connect=30.0, read=30.0, write=30.0, pool=30.0)
        limits = httpx.Limits(max_connections=20, max_keepalive_connections=20)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Screaming Frog SEO Spider/20.0; +http://www.screamingfrog.co.uk/seo-spider/)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "es-ES,es;q=0.9",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        app.state.client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            headers=headers,
            http1=True,
            follow_redirects=True
        )
    except Exception as e:
        print(f"❌ Error creating AsyncClient: {e}", file=sys.stderr)
        raise RuntimeError("Failed to initialize Async HTTP Client")

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
    batch_size: int = 10  # valor por defecto
    start: int = 0        # valor por defecto para iniciar, asegura que siempre tenga un valor

# Definición del modo de rastreo global
crawler_mode = {
    "safe_mode": False,
    "error_counter": 0,
    "current_domain": None
}

def get_dynamic_headers():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.128 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36",
        "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        "Mozilla/5.0 (compatible; Screaming Frog SEO Spider/20.0; +http://www.screamingfrog.co.uk/seo-spider/)"
    ]

    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
        "Upgrade-Insecure-Requests": "1",
    }
    return headers

async def safe_analyze(url, client, semaphore):
    domain = urlparse(url).netloc

    if crawler_mode["current_domain"] != domain:
        crawler_mode["current_domain"] = domain
        crawler_mode["error_counter"] = 0
        crawler_mode["safe_mode"] = False
        crawler_mode["concurrency"] = 10
        print(f"🌐 Nuevo dominio detectado: {domain}. Reiniciando contadores.")

    concurrency = crawler_mode.get("concurrency", 10)
    sleep_between_requests = random.uniform(0.6, 0.9) if not crawler_mode["safe_mode"] else random.uniform(3.5, 6.0)
    max_retries = 5

    async with semaphore:
        try:
            dynamic_headers = get_dynamic_headers()
            timeout_per_url = 15  # ⏱️ Máximo 15 segundos por URL
            try:
                result, error_type = await asyncio.wait_for(
                    retry_analyze_url(url, client, max_retries=max_retries, custom_headers=dynamic_headers),
                    timeout=timeout_per_url
                )
            except asyncio.TimeoutError:
                print(f"⏰ Timeout individual: {url} tardó más de {timeout_per_url} segundos. Saltando URL.")
                crawler_mode["error_counter"] += 1
                return None

            if not result:
                if error_type != "502":
                    crawler_mode["error_counter"] += 1
                    print(f"⚠️ Error acumulado ({crawler_mode['error_counter']}) en {domain}. Error detectado: {error_type}")
            else:
                crawler_mode["error_counter"] = 0

            if error_type in ["429", "503", "network_or_http", "unknown"]:
                crawler_mode["safe_mode"] = True
                crawler_mode["concurrency"] = 1
                print(f"🚨 Server unstable detected. Switching to SAFE MODE para {domain} (por {error_type}).")

            if crawler_mode["error_counter"] >= 3:
                crawler_mode["safe_mode"] = True
                crawler_mode["concurrency"] = 1

            await asyncio.sleep(sleep_between_requests)
            return result

        except Exception as e:
            crawler_mode["error_counter"] += 1
            print(f"❌ Excepción grave analizando {url}: {e} | Error count: {crawler_mode['error_counter']}")
            if crawler_mode["error_counter"] >= 3:
                crawler_mode["safe_mode"] = True
                crawler_mode["concurrency"] = 1
            await asyncio.sleep(sleep_between_requests)
            return None

@app.post("/start_batch")
async def start_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    batch_id = f"batch_{int(time.time())}"  # 🔥 batch ID único basado en timestamp
    batch_results[batch_id] = {
        "status": "processing",
        "processed_urls": [],
        "start_index": request.start
    }
    background_tasks.add_task(process_batch_background, batch_id, request)
    return {"message": "Batch started", "batch_id": batch_id}

async def process_batch_background(batch_id: str, request: BatchRequest):
    domain = request.domain.rstrip('/')
    print(f"🔎 (Background) Fetching URLs from domain: {domain}")

    urls = await fetch_sitemap(app.state.client, domain) or []

    if not urls:
        batch_results[batch_id]["status"] = "failed"
        return

    urls_to_process = urls[request.start:request.start + request.batch_size]

    if not urls_to_process:
        batch_results[batch_id]["status"] = "done"
        return

    concurrency = crawler_mode.get("concurrency", 10)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [safe_analyze(url, app.state.client, semaphore) for url in urls_to_process]
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

    batch_results[batch_id]["processed_urls"] = valid_results
    batch_results[batch_id]["status"] = "done"

@app.get("/get_batch_result/{batch_id}")
async def get_batch_result(batch_id: str):
    if batch_id not in batch_results:
        raise HTTPException(status_code=404, detail="Batch ID not found")

    return batch_results[batch_id]

@app.post("/full_process_domain")
async def full_process_domain(request: BatchRequest):
    domain = request.domain.rstrip('/')
    batch_size = request.batch_size
    start = request.start

    print(f"🚀 Iniciando procesamiento completo de {domain}")

    all_processed_urls = []

    while True:
        print(f"🔎 Procesando batch desde índice {start}")

        batch_request = BatchRequest(
            domain=domain,
            batch_size=batch_size,
            start=start
        )
        response = await process_urls_in_batches(batch_request)

        batch_processed = response.get("processed_urls", [])
        more_batches = response.get("more_batches", False)
        next_start = response.get("next_batch_start", 0)

        all_processed_urls.extend(batch_processed)

        if not more_batches:
            print("✅ Todos los batches procesados.")
            break

        start = next_start  # Preparar el siguiente batch

    return {
        "total_urls_processed": len(all_processed_urls),
        "processed_urls": all_processed_urls
    }

@app.post("/process_urls_in_batches")
async def process_urls_in_batches(request: BatchRequest):
    domain = request.domain.rstrip('/')
    print(f"🔎 Fetching URLs from domain: {domain}")

    urls = await fetch_sitemap(app.state.client, domain) or []

    if not urls:
        print("🚫 No URLs found in sitemap.")
        return {"processed_urls": [], "more_batches": False, "next_batch_start": 0}

    urls_to_process = urls[request.start:request.start + request.batch_size]

    if not urls_to_process:
        print("🚫 No URLs to process in this batch.")
        return {"processed_urls": [], "more_batches": False, "next_batch_start": 0}

    concurrency = crawler_mode.get("concurrency", 10)
    semaphore = asyncio.Semaphore(concurrency)

    timeout_batch = 45  # ⏱️ Máximo 45 segundos para todo el batch
    try:
        tasks = [safe_analyze(url, app.state.client, semaphore) for url in urls_to_process]
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout_batch)
    except asyncio.TimeoutError:
        print(f"⏰ Timeout de batch: más de {timeout_batch} segundos. Cortando procesamiento.")
        results = []

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
            print(f"🌎 Intento {attempt}: Fetching {url}")
            response = await client.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            print(f"✅ Success: {url}")
            return response
        except (httpx.RequestError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            print(f"⚠️ Error {e} in {url}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
        except Exception as e:
            print(f"❌ Fatal error fetching {url}: {e}")
            break
    print(f"🛑 Failed after {retries} attempts: {url}")
    return None

async def fetch_sitemap(client: httpx.AsyncClient, base_url: str):
    headers = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.184 Mobile Safari/537.36 "
                  "(compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
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

async def retry_analyze_url(url: str, client: httpx.AsyncClient, max_retries: int = 5, initial_delay: float = 1.0, custom_headers: dict = None):
    delay = initial_delay
    last_error_type = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"🔄 Attempt {attempt} fetching: {url}")
            result = await analyze_url(url, client, headers=custom_headers)
            if result:
                return result, None
            else:
                print(f"⚠️ Empty result en {url} attempt {attempt}")

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            print(f"⚠️ HTTPStatusError ({status_code}) en {url}: {e}")
            if status_code == 502:
                return None, "502"
            elif status_code in [429, 503]:
                last_error_type = str(status_code)
            else:
                last_error_type = "http_error"

        except (httpx.RequestError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_error_type = "network_or_http"
            print(f"⚠️ Error de red en {url}: {e}")

        except Exception as e:
            last_error_type = "unknown"
            print(f"❌ Unexpected error en {url}: {e}")

        await asyncio.sleep(delay)
        delay *= random.uniform(1.2, 1.5)

    print(f"🛑 Failed fetching {url} after {max_retries} retries.")
    return None, last_error_type


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
        
        except RequestException as e:
            attempts += 1
            sleep_time = (2 ** attempts) * 5
            time.sleep(sleep_time * random.uniform(1.5, 2))  # CORREGIDO: sleep sin async
            
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