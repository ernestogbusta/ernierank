# Este es el archivo main.py, dame código robusto y completo para solucionar el problema pero dame ÚNICAMENTE las funciones o endpoints que debo modificar o actualizar, EN NINGUN CASO me des funciones o endpoints que ya estén funcionando bien en mi código

from analyze_url import analyze_url
from analyze_internal_links import analyze_internal_links, InternalLinkAnalysis, correct_url_format
from analyze_wpo import analyze_wpo
from analyze_cannibalization import analyze_cannibalization
from analyze_thin_content import analyze_thin_content, fetch_processed_data_or_process_batches, calculate_thin_content_score_and_details, clean_and_split, classify_content_level
from generate_content import generate_seo_content, process_new_data
from analyze_404 import fetch_urls, check_url, crawl_site, find_broken_links
from fastapi import FastAPI, HTTPException, Depends, Body, Request, BackgroundTasks
import httpx
from httpx import AsyncClient, Timeout, RemoteProtocolError
from bs4 import BeautifulSoup
import xmltodict
import os
import json
from pydantic import BaseModel, HttpUrl, validator
import uvicorn
from collections import Counter
from typing import List, Dict, Optional
from urllib.parse import urlparse, urljoin
import urllib.parse
import re
import asyncio
import time
import requests
import logging
from starlette.middleware.gzip import GZipMiddleware

app = FastAPI(title="ErnieRank API")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@app.on_event("startup")
async def startup_event():
    logging.basicConfig(level=logging.INFO)
    app.state.openai_api_key = os.getenv("OPENAI_API_KEY")
    if not app.state.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment variables")
    app.state.client = httpx.AsyncClient()

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.client.aclose()

# Configuración del logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("CannibalizationAnalysis")

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
    start: int = 0        # valor por defecto para iniciar

@app.post("/process_urls_in_batches")
async def process_urls_in_batches(request: BatchRequest):
    sitemap_url = f"{request.domain.rstrip('/')}/sitemap.xml"  # Comienza intentando con la ubicación más común
    client = httpx.AsyncClient(follow_redirects=True)  # Asegúrate de seguir las redirecciones automáticamente
    urls = await fetch_sitemap(client, sitemap_url)

    if not urls:
        print("No URLs found in the sitemap.")
        raise HTTPException(status_code=404, detail="Sitemap not found or empty")
    
    print(f"Total URLs fetched for processing: {len(urls)}")
    urls_to_process = urls[request.start:request.start + request.batch_size]
    print(f"URLs to process from index {request.start} to {request.start + request.batch_size}: {urls_to_process}")

    tasks = [analyze_url(url, client) for url in urls_to_process]
    results = await asyncio.gather(*tasks)
    await client.aclose()

    print(f"Results received: {results}")

    valid_results = [
        {
            "url": result['url'],
            "title": result.get('title', "No title provided"),
            "meta_description": result.get('meta_description', "No description provided"),
            "main_keyword": result.get('main_keyword', "Not specified"),
            "secondary_keywords": result.get('secondary_keywords', []),
            "semantic_search_intent": result.get('semantic_search_intent', "Not specified")
        } for result in results if result
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

async def fetch_sitemap(client, url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        "Accept": "application/xml, application/xhtml+xml, text/html, application/json; q=0.9, */*; q=0.8"
    }
    try:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        sitemap_contents = xmltodict.parse(response.content)
        all_urls = []

        # Manejo dinámico de sitemap index y urlset
        if 'sitemapindex' in sitemap_contents:
            # Asegura que sitemap_indices siempre es una lista
            sitemap_indices = sitemap_contents['sitemapindex'].get('sitemap')
            if isinstance(sitemap_indices, list):
                for sitemap in sitemap_indices:
                    sitemap_url = sitemap['loc']
                    all_urls.extend(await fetch_sitemap(client, sitemap_url))
            else:
                sitemap_url = sitemap_indices['loc']
                all_urls.extend(await fetch_sitemap(client, sitemap_url))
        elif 'urlset' in sitemap_contents:
            # Asegura que los URLs siempre son manejados como lista
            url_entries = sitemap_contents['urlset']['url']
            if isinstance(url_entries, list):
                all_urls = [url['loc'] for url in url_entries]
            else:
                all_urls = [url_entries['loc']]

        print(f"Fetched {len(all_urls)} URLs from the sitemap at {url}.")
        return all_urls
    except Exception as e:
        print(f"Error fetching or parsing sitemap from {url}: {str(e)}")
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
    logging.debug(f"Request received: {await request.json()}")
    req_data = await request.json()
    url = req_data.get("url")

    if not url:
        logging.error("URL not provided in the request")
        raise HTTPException(status_code=422, detail="URL parameter is required.")

    try:
        new_data = await process_new_data(url, app.state.client)
        if not new_data:
            logging.error(f"No data could be processed from the URL: {url}")
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
        logging.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
        raise HTTPException(status_code=exc.response.status_code, detail=f"HTTP error: {exc.response.text}")
    except httpx.RequestError as exc:
        logging.error(f"An error occurred while making HTTP call to OpenAI: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"HTTP request failed: {str(exc)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


##############################################


########### ANALYZE_THIN_CONTENT ##########

# Configurando el logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    processed_urls: List[PageData]
    more_batches: bool = False
    next_batch_start: Optional[int] = None

    @validator('processed_urls', each_item=True)
    def check_urls(cls, v):
        if not v.title or not v.url:
            logging.error(f"Validation error for URL data: {v}")
            raise ValueError("URL and title must be provided for each item.")
        return v

@app.post("/analyze_thin_content")
async def analyze_thin_content_endpoint(request: Request):
    try:
        request_data = await request.json()
        logging.debug(f"Raw request data: {request_data}")

        try:
            thin_request = ThinContentRequest(**request_data)
        except Exception as e:
            logging.error(f"Error parsing request data: {e}, Data received: {request_data}")
            raise HTTPException(status_code=400, detail=f"Error parsing request data: {e}")

        logging.debug(f"Request parsed successfully with data: {thin_request}")

        if not thin_request.processed_urls:
            logging.error("No URLs provided in the request.")
            raise HTTPException(status_code=400, detail="No URLs provided")

        logging.info("Starting thin content analysis.")
        analysis_results = await analyze_thin_content(thin_request)
        logging.info(f"Analysis results obtained: {analysis_results}")

        formatted_response = {
            "thin_content_pages": [
                {
                    "url": urllib.parse.urlparse(page["url"]).path,  # Devuelve solo la parte del path de la URL
                    "level": page["level"],
                    "description": page["details"]
                }
                for page in analysis_results["thin_content_pages"]
            ]
        }
        logging.info(f"Formatted response ready to be sent: {formatted_response}")

        return formatted_response

    except HTTPException as http_exc:
        # Log specific for HTTP errors that are raised deliberately
        logging.error(f"HTTP error during request processing: {http_exc.detail}")
        raise
    except Exception as e:
        logging.critical(f"Unexpected error during request processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def tarea_demorada(nombre: str):
    logging.debug(f"Iniciando tarea demorada para {nombre}")
    time.sleep(10)  # Simula un proceso que tarda 10 segundos
    logging.debug(f"Tarea {nombre} completada")

@app.post("/start-delayed-task/")
async def start_delayed_task(nombre: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(tarea_demorada, nombre=nombre)
    logging.info(f"Tarea demorada iniciada en segundo plano para {nombre}")
    return {"message": "Tarea demorada iniciada en segundo plano"}

def analyze_content_in_background(request: ThinContentRequest):
    logging.debug("Iniciando análisis de contenido delgado en segundo plano")
    for page in request.processed_urls:
        logging.debug(f"Analizando en segundo plano {page.url}")
    logging.debug("Análisis de contenido delgado en segundo plano completado")

async def analyze_thin_content_directly(request: ThinContentRequest):
    if not request.processed_urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    results = []
    for page in request.processed_urls:
        try:
            score, level, details = await calculate_thin_content_score_and_details(page)
            results.append({
                "url": page.url,
                "score": score,
                "level": level,
                "details": details
            })
            logging.debug(f"Processed {page.url} with score {score}, level {level}.")
        except Exception as e:
            logging.error(f"Failed to process {page.url}: {str(e)}")
            continue
    return {"message": "Análisis completado", "data": results}

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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="debug")