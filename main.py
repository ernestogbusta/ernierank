# main.py

from analyze_url import analyze_url
from analyze_internal_links import analyze_internal_links, InternalLinkAnalysis, correct_url_format
from analyze_wpo import analyze_wpo
from analyze_cannibalization import analyze_cannibalization
from analyze_thin_content import analyze_thin_content, fetch_processed_data_or_process_batches, calculate_thin_content_score_and_details, clean_and_split, classify_content_level
from generate_content import generate_seo_content, process_new_data
from fastapi import FastAPI, HTTPException, Request, Body, BackgroundTasks
import httpx
from bs4 import BeautifulSoup
import xmltodict
import os
import json
from pydantic import BaseModel, HttpUrl
import uvicorn
from collections import Counter
from typing import List, Dict, Optional
from urllib.parse import urlparse
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

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Configuración del logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("CannibalizationAnalysis")

########## ANALYZE_URL ############

class BatchRequest(BaseModel):
    domain: str
    batch_size: int = 100  # valor por defecto
    start: int = 0        # valor por defecto para iniciar, asegura que siempre tenga un valor

@app.post("/process_urls_in_batches")
async def process_urls_in_batches(request: BatchRequest):
    sitemap_url = f"{request.domain.rstrip('/')}/sitemap_index.xml"
    print(f"Fetching URLs from: {sitemap_url}")
    urls = await fetch_sitemap(app.state.client, sitemap_url)

    if not urls:
        print("No URLs found in the sitemap.")
        raise HTTPException(status_code=404, detail="Sitemap not found or empty")
    
    print(f"Total URLs fetched for processing: {len(urls)}")
    urls_to_process = urls[request.start:request.start + request.batch_size]
    print(f"URLs to process from index {request.start} to {request.start + request.batch_size}: {urls_to_process}")

    tasks = [analyze_url(url, app.state.client) for url in urls_to_process]
    results = await asyncio.gather(*tasks)
    print(f"Results received: {results}")

    # Cambio en el filtro para permitir resultados con main_keyword o secondary_keywords vacíos
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

        if 'sitemapindex' in sitemap_contents:
            sitemap_indices = sitemap_contents['sitemapindex']['sitemap']
            sitemap_indices = sitemap_indices if isinstance(sitemap_indices, list) else [sitemap_indices]
            for sitemap in sitemap_indices:
                sitemap_url = sitemap['loc']
                sitemap_resp = await client.get(sitemap_url, headers=headers)
                sitemap_resp.raise_for_status()
                individual_sitemap = xmltodict.parse(sitemap_resp.content)
                urls = [url['loc'] for url in individual_sitemap['urlset']['url']]
                all_urls.extend(urls)
        elif 'urlset' in sitemap_contents:
            all_urls = [url['loc'] for url in sitemap_contents['urlset']['url']]

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
    url: str
    title: str
    meta_description: str
    h1: Optional[str] = None
    h2: Optional[List[str]] = []
    main_keyword: str
    secondary_keywords: List[str]
    semantic_search_intent: str

class ThinContentRequest(BaseModel):
    processed_urls: List[PageData]

def tarea_demorada(nombre: str):
    print(f"Iniciando tarea para {nombre}")
    time.sleep(10)  # Simula un proceso que tarda 10 segundos
    print(f"Tarea completada para {nombre}")

@app.post("/start-delayed-task/")
async def start_delayed_task(nombre: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(tarea_demorada, nombre=nombre)
    return {"message": "Tarea demorada iniciada en segundo plano"}

def analyze_content_in_background(request: ThinContentRequest):
    print("Iniciando análisis de contenido delgado...")
    for page in request.processed_urls:
        print(f"Analizando {page.url}...")
    print("Análisis de contenido delgado completado.")

@app.post("/analyze_thin_content")
async def analyze_thin_content(request: ThinContentRequest):
    if not request.processed_urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    
    # Realiza el análisis directamente aquí y espera a que se complete.
    thin_content_results = await analyze_thin_content_directly(request)
    return thin_content_results

async def analyze_thin_content_directly(request: ThinContentRequest):
    results = []
    for page in request.processed_urls:
        score, details = await calculate_thin_content_score_and_details(page)  # Asumiendo que esta función es asíncrona
        level = classify_content_level(score)  # Clasifica el nivel basado en el score
        results.append({
            "url": page.url,
            "thin content level": level,
            "details": details
        })
    return {"message": "Análisis completado", "data": results}

#######################################



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="debug")