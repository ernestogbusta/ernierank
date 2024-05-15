# Este es el archivo main.py, dame código robusto y completo para solucionar el problema pero dame ÚNICAMENTE las funciones o endpoints que debo modificar o actualizar, EN NINGUN CASO me des funciones o endpoints que ya estén funcionando bien en mi código

from analyze_url import analyze_url
from analyze_internal_links import analyze_internal_links, InternalLinkAnalysis, correct_url_format
from analyze_wpo import analyze_wpo
from analyze_cannibalization import analyze_cannibalization
from analyze_thin_content import analyze_thin_content, fetch_processed_data_or_process_batches, calculate_thin_content_score_and_details, clean_and_split, classify_content_level
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
from typing import List, Dict, Optional
from urllib.parse import urlparse, urljoin
import urllib.parse
import re
import asyncio
import time
import requests
import logging
from starlette.middleware.gzip import GZipMiddleware
import pytrends
from pytrends.request import TrendReq
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import sys

app = FastAPI(title="ErnieRank API")

@app.on_event("startup")
async def startup_event():
    app.state.openai_api_key = os.getenv("OPENAI_API_KEY")
    if not app.state.openai_api_key:
        logger.critical("OPENAI_API_KEY environment variable not set. Set this variable and restart the application.")
        print("Failed to detect OPENAI_API_KEY:", app.state.openai_api_key, file=sys.stderr)
        raise RuntimeError("OPENAI_API_KEY is not set in the environment variables")
    else:
        print("OPENAI_API_KEY detected successfully:", app.state.openai_api_key, file=sys.stderr)
    app.state.client = httpx.AsyncClient()

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
    # Corrigiendo la URL para asegurar que siempre sea HTTPS
    base_url = f"https://{urlparse(request.domain).netloc}"
    sitemap_url = f"{base_url}/sitemap_index.xml"
    print(f"Fetching URLs from: {sitemap_url}")

    urls = await fetch_sitemap(app.state.client, base_url)

    if not urls:
        print("No sitemap XML found at any known locations.")
        raise HTTPException(status_code=404, detail="Sitemap not found or empty")

    print(f"Total URLs fetched for processing: {len(urls)}")
    urls_to_process = urls[request.start:request.start + request.batch_size]
    print(f"URLs to process from index {request.start} to {request.start + request.batch_size}: {urls_to_process}")

    tasks = [analyze_url(url, app.state.client) for url in urls_to_process]
    results = await asyncio.gather(*tasks)
    print(f"Results received: {results}")

    # Añadir aviso si la URL no usa HTTPS
    extended_results = []
    for result in results:
        if result is not None:
            result['https_warning'] = "URL is not using HTTPS." if not result['url'].startswith("https://") else ""
            extended_results.append(result)
        else:
            print("Analysis for a URL failed or returned no data.")

    print(f"Filtered results: {extended_results}")

    next_start = request.start + len(urls_to_process)
    more_batches = next_start < len(urls)
    print(f"More batches: {more_batches}, Next batch start index: {next_start}")

    return {
        "processed_urls": extended_results,
        "more_batches": more_batches,
        "next_batch_start": next_start if more_batches else None
    }

async def fetch_sitemap(client, base_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        "Accept": "application/xml, application/xhtml+xml, text/html, application/json; q=0.9, */*; q=0.8"
    }
    # Asegurarse de que la base URL es correcta, eliminando cualquier ruta adicional incorrectamente añadida
    base_url = urlparse(base_url).scheme + "://" + urlparse(base_url).netloc

    sitemap_paths = ['/sitemap_index.xml', '/sitemap.xml', '/sitemap1.xml']  # Diferentes endpoints de sitemap comunes
    all_urls = []

    for path in sitemap_paths:
        url = f"{base_url.rstrip('/')}{path}"
        try:
            response = await client.get(url, headers=headers)
            if response.status_code == 404:
                continue  # Si no se encuentra el sitemap en esta ruta, intenta con la siguiente
            response.raise_for_status()
            sitemap_contents = xmltodict.parse(response.content)

            # Procesando sitemap index
            if 'sitemapindex' in sitemap_contents:
                sitemap_indices = sitemap_contents['sitemapindex'].get('sitemap', [])
                sitemap_indices = sitemap_indices if isinstance(sitemap_indices, list) else [sitemap_indices]
                for sitemap in sitemap_indices:
                    sitemap_url = sitemap['loc']
                    all_urls.extend(await fetch_individual_sitemap(client, sitemap_url))
            # Procesando urlset directamente si está presente
            elif 'urlset' in sitemap_contents:
                all_urls.extend([url['loc'] for url in sitemap_contents['urlset']['url']])
        except Exception as e:
            print(f"Error fetching or parsing sitemap from {url}: {str(e)}")

    if not all_urls:
        print("No sitemaps found at any known locations.")
        return None
    return all_urls

async def fetch_individual_sitemap(client, sitemap_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        "Accept": "application/xml, application/xhtml+xml, text/html, application/json; q=0.9, */*; q=0.8"
    }
    try:
        response = await client.get(sitemap_url, headers=headers)
        response.raise_for_status()
        sitemap_contents = xmltodict.parse(response.content)
        if 'urlset' in sitemap_contents:
            return [url['loc'] for url in sitemap_contents['urlset']['url']]
    except Exception as e:
        print(f"Error fetching or parsing individual sitemap from {sitemap_url}: {str(e)}")
        return []

    return []

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
    processed_urls: List[PageData]
    more_batches: bool = False
    next_batch_start: Optional[int] = None

    @validator('processed_urls', each_item=True)
    def check_urls(cls, v):
        if not v.title or not v.url:
            raise ValueError("URL and title must be provided for each item.")
        return v

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

        analysis_results = await analyze_thin_content(thin_request)

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

        return formatted_response

    except HTTPException as http_exc:
        # Log specific for HTTP errors that are raised deliberately
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def tarea_demorada(nombre: str):
    time.sleep(10)  # Simula un proceso que tarda 10 segundos

@app.post("/start-delayed-task/")
async def start_delayed_task(nombre: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(tarea_demorada, nombre=nombre)
    return {"message": "Tarea demorada iniciada en segundo plano"}

def analyze_content_in_background(request: ThinContentRequest):
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
        except Exception as e:
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

class KeywordRequest(BaseModel):
    topic: str

async def fetch_google_search_results(topic, max_attempts=5):
    attempts = 0
    while attempts < max_attempts:
        try:
            pytrends.build_payload([topic])
            related_queries = pytrends.related_queries()
            return process_related_queries(related_queries, topic)
        except Exception as e:
            attempts += 1
            sleep_time = (2 ** attempts) * 5  # Tiempo de espera exponencial con base 5 segundos
            await asyncio.sleep(sleep_time)
            if attempts == max_attempts:
                raise HTTPException(status_code=429, detail="Google Trends rate limit exceeded")

@app.post("/search_keywords")
async def search_keywords(request: KeywordRequest):
    google_results = await fetch_google_search_results(request.topic)
    if not google_results:
        raise HTTPException(status_code=404, detail="No keywords found")
    return {"keywords": google_results}

def process_related_queries(related_queries, topic):
    keywords = []
    scaling_factor = 14800 / 100  # Suponiendo que 100 es el máximo de Trends para 'Digital Marketing'
    excluded_words = ['la', 'el', 'los', 'las']

    if topic in related_queries:
        related_data = related_queries[topic]
        for data_type in ['top', 'rising']:
            if related_data.get(data_type) is not None:
                for query in related_data[data_type].to_dict('records'):
                    if not any(excluded_word in query['query'].split() for excluded_word in excluded_words):
                        trends_volume = query['value'] * scaling_factor
                        scaled_volume = adjust_volume(trends_volume)
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