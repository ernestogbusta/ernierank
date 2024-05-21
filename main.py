# Este es el archivo main.py, dame código robusto y completo para solucionar el problema pero dame ÚNICAMENTE las funciones o endpoints que debo modificar o actualizar, EN NINGUN CASO me des funciones o endpoints que ya estén funcionando bien en mi código

from analyze_url import analyze_url
from analyze_internal_links import analyze_internal_links, InternalLinkAnalysis, correct_url_format
from analyze_wpo import analyze_wpo
from analyze_cannibalization import analyze_cannibalization
from analyze_thin_content import calculate_thin_content_score_and_details, classify_content_level
from generate_content import generate_seo_content, process_new_data
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
    batch_size: int = 100
    start: int = 0

@app.post("/process_urls_in_batches")
async def process_urls_in_batches(request: BatchRequest):
    sitemap_url = f"{request.domain.rstrip('/')}/sitemap_index.xml"
    print(f"Fetching URLs from: {sitemap_url}")
    urls = await fetch_sitemap(sitemap_url)

    if not urls:
        print("No URLs found in the sitemap.")
        raise HTTPException(status_code=404, detail="Sitemap not found or empty")
    
    print(f"Total URLs fetched for processing: {len(urls)}")
    urls_to_process = urls[request.start:request.start + request.batch_size]
    print(f"URLs to process from index {request.start} to {request.start + request.batch_size}: {urls_to_process}")

    async with httpx.AsyncClient() as client:
        tasks = [analyze_url(url, client) for url in urls_to_process]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    print(f"Results received: {results}")

    valid_results = [
        {
            "url": result['url'],
            "title": result.get('title', "No title provided"),
            "meta_description": result.get('meta_description', "No description provided"),
            "main_keyword": result.get('main_keyword', "Not specified"),
            "secondary_keywords": result.get('secondary_keywords', []),
            "semantic_search_intent": result.get('semantic_search_intent', "Not specified")
        } for result in results if result and not isinstance(result, Exception)
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

async def fetch_sitemap(base_url: str) -> List[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        "Accept": "application/xml, application/xhtml+xml, text/html, application/json; q=0.9, */*; q=0.8"
    }
    base_url = urlparse(base_url).scheme + "://" + urlparse(base_url).netloc

    sitemap_paths = ['/sitemap_index.xml', '/sitemap.xml', '/sitemap1.xml']
    all_urls = []

    async with httpx.AsyncClient() as client:
        for path in sitemap_paths:
            url = f"{base_url.rstrip('/')}{path}"
            try:
                response = await client.get(url, headers=headers)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                sitemap_contents = xmltodict.parse(response.text)

                if 'sitemapindex' in sitemap_contents:
                    sitemap_indices = sitemap_contents['sitemapindex'].get('sitemap', [])
                    sitemap_indices = sitemap_indices if isinstance(sitemap_indices, list) else [sitemap_indices]
                    for sitemap in sitemap_indices:
                        sitemap_url = sitemap['loc']
                        all_urls.extend(await fetch_individual_sitemap(sitemap_url, client))
                elif 'urlset' in sitemap_contents:
                    all_urls.extend([url['loc'] for url in sitemap_contents['urlset']['url']])
            except Exception as e:
                print(f"Error fetching or parsing sitemap from {url}: {str(e)}")

    # Filtrar y sanitizar URLs
    sanitized_urls = [sanitize_url(url) for url in all_urls if is_valid_url(url)]
    if not sanitized_urls:
        print("No valid URLs found after sanitization.")
        return []
    return sanitized_urls

async def fetch_individual_sitemap(sitemap_url: str, client: httpx.AsyncClient) -> List[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, como Gecko) Chrome/58.0.3029.110 Safari/537.36",
        "Accept": "application/xml, application/xhtml+xml, text/html, application/json; q=0.9, */*; q=0.8"
    }
    try:
        response = await client.get(sitemap_url, headers=headers)
        response.raise_for_status()
        sitemap_contents = xmltodict.parse(response.text)
        if 'urlset' in sitemap_contents:
            return [url['loc'] for url in sitemap_contents['urlset']['url']]
    except Exception as e:
        print(f"Error fetching or parsing individual sitemap from {sitemap_url}: {str(e)}")
        return []

    return []

def sanitize_url(url: str) -> str:
    """
    Sanitiza la URL removiendo caracteres no válidos al final de la misma.
    """
    return url.rstrip(':')

def is_valid_url(url: str) -> bool:
    """
    Verifica si una URL es válida.
    """
    parsed_url = urlparse(url)
    return parsed_url.scheme in ["http", "https"] and bool(parsed_url.netloc) and not url.endswith(':')

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

@app.post("/analyze_thin_content/")
async def analyze_thin_content_endpoint(request: ThinContentRequest):
    if not request.processed_urls:
        raise HTTPException(status_code=404, detail="No URL data available for analysis.")

    logging.debug("Inicio del análisis de contenido delgado")
    
    try:
        tasks = [calculate_thin_content_score_and_details(page) for page in request.processed_urls]
        results = await asyncio.gather(*tasks)
    except Exception as e:
        logging.error(f"Error durante el análisis de contenido delgado: {e}")
        raise HTTPException(status_code=500, detail="Error processing thin content analysis")

    try:
        thin_content_pages = [
            {
                "url": page.url,
                "level": classify_content_level(result[0]),
                "description": result[1]
            }
            for page, result in zip(request.processed_urls, results) if classify_content_level(result[0]) != "none"
        ]
    except Exception as e:
        logging.error(f"Error durante la clasificación de contenido: {e}")
        raise HTTPException(status_code=500, detail="Error classifying thin content levels")

    logging.debug("Fin del análisis de contenido delgado")

    return {"thin_content_pages": thin_content_pages} if thin_content_pages else {"message": "No thin content detected"}

@app.post("/analyze_domain/")
async def analyze_domain(domain: str, batch_size: int = 10):
    start_index = 0
    all_thin_content_pages = []

    while True:
        batch_request = URLBatchRequest(domain=domain, batch_size=batch_size, start_index=start_index)
        batch_response = await process_urls_in_batches(batch_request)

        thin_content_request = ThinContentRequest(
            processed_urls=[PageData(**url) for url in batch_response["processed_urls"]],
            more_batches=batch_response["more_batches"],
            next_batch_start=batch_response["next_batch_start"]
        )
        
        thin_content_response = await analyze_thin_content_endpoint(thin_content_request)
        
        if "thin_content_pages" in thin_content_response:
            all_thin_content_pages.extend(thin_content_response["thin_content_pages"])

        if not batch_response["more_batches"]:
            break

        start_index = batch_response["next_batch_start"]

    return {"thin_content_pages": all_thin_content_pages} if all_thin_content_pages else {"message": "No thin content detected"}


#######################################



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