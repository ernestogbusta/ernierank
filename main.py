# main.py

import cProfile
import pstats
import io
from analyze_url import analyze_url
from analyze_internal_links import analyze_internal_links, InternalLinkAnalysis, correct_url_format
from analyze_wpo import analyze_wpo
from analyze_cannibalization import analyze_cannibalization
from fastapi import FastAPI, HTTPException, Request, Body
import httpx
from bs4 import BeautifulSoup
import xmltodict
import os
import json
from pydantic import BaseModel
import uvicorn
from collections import Counter
from typing import List, Dict, Optional
from urllib.parse import urlparse
import asyncio
import time
import requests
import logging

# Configuración del logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("CannibalizationAnalysis")

app = FastAPI(title="ErnieRank API")

@app.middleware("http")
async def log_process_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    response.headers["X-Process-Time"] = str(duration)
    logger.info(f"Request path: {request.url.path}, Duration: {duration:.2f} seconds")
    return response

@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc} - URL: {request.url}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )

# Punto de entrada que maneja redirecciones y captura de errores de red
@app.get("/redirect/{url:path}")
async def handle_redirect(url: str):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            response = await client.get(url)
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Not Found")
            return {"URL Final": str(response.url), "Status": response.status_code}
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Request Error: {str(e)}")

# Manejador para la raíz que simplemente verifica la conectividad
@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.on_event("startup")
async def startup_event():
    app.state.client = httpx.AsyncClient()

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.client.aclose()

# Simulación de un endpoint para manejar errores
@app.get("/error")
async def error_handler():
    return {"message": "There was an error with your request."}

class URLData(BaseModel):
    title: str
    h1: str
    main_keyword: str
    secondary_keywords: List[str]
    semantic_search_intent: str

@app.post("/test")
async def test_logging(data: List[URLData]):
    logger.debug(f"Received data: {data}")
    if not data:
        logger.info("No data provided.")
        return {"message": "No data provided."}
    else:
        # Simulate processing
        logger.info("Processing data...")
        return {"message": "Data processed."}


class BatchRequest(BaseModel):
    domain: str
    batch_size: int = 100  # valor por defecto
    start: int = 0        # valor por defecto para iniciar, asegura que siempre tenga un valor

@app.on_event("startup")
async def startup_event():
    app.state.client = httpx.AsyncClient()
    app.state.progress_file = "progress.json"
    if not os.path.exists(app.state.progress_file):
        with open(app.state.progress_file, 'w') as file:
            json.dump({"current_index": 0, "urls": []}, file)

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.client.aclose()


########## ANALYZE_URL ############

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


class URLData(BaseModel):
    url: str
    title: str
    meta_description: str
    main_keyword: str
    secondary_keywords: List[str]
    semantic_search_intent: str

class CannibalizationRequest(BaseModel):
    processed_urls: List[URLData]
    more_batches: Optional[bool] = False
    next_batch_start: Optional[int] = None

@app.post("/analyze_cannibalization")
async def analyze_cannibalization_endpoint(request: CannibalizationRequest):
    start_time = time.time()
    try:
        from analyze_cannibalization import analyze_cannibalization  # Ensure to import correctly
        results = await analyze_cannibalization(request.processed_urls)
        duration = time.time() - start_time
        logger.info(f"Analysis completed in {duration:.2f} seconds")
        return results
    except HTTPException as http_exc:
        logger.warning(f"HTTP error during cannibalization analysis: {http_exc.detail}")
        raise
    except Exception as exc:
        logger.error(f"Error during cannibalization analysis: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


##############################################



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="debug")

