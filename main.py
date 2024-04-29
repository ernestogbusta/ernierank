# main.py

from analyze_url import analyze_url
from analyze_internal_links import analyze_internal_links, InternalLinkAnalysis, correct_url_format
from analyze_wpo import analyze_wpo
from analyze_cannibalization import analyze_cannibalization, CannibalizationURLData, analyze_url_for_cannibalization
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
import re
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

class URLData(BaseModel):
    url: str
    title: str
    meta_description: str
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


# Definición de modelos para el análisis de canibalización
class CannibalizationRequest(BaseModel):
    processed_urls: List[URLData]
    more_batches: Optional[bool] = False
    next_batch_start: Optional[int] = None

@app.post("/analyze_cannibalization")
async def analyze_cannibalization_endpoint(domain: str):
    # Crear un BatchRequest con el dominio y configuraciones de lote
    request = BatchRequest(domain=domain, batch_size=100, start=0)
    response = await process_urls_in_batches_for_cannibalization(request)
    cannibalization_urls = response.get('processed_urls', [])  # Cambiando el nombre de la variable para claridad
    
    # Convertir URLData a CannibalizationURLData antes del análisis
    cannibalization_data = convert_to_cannibalization_data(cannibalization_urls)
    results = analyze_cannibalization(cannibalization_data)
    if results.get("message"):
        return {"message": results["message"]}
    else:
        return {"cannibalization_issues": results}

def convert_to_cannibalization_data(url_data_list: List[Dict]) -> List[CannibalizationURLData]:
    """Converts a list of URLData dictionaries to CannibalizationURLData by extracting necessary fields."""
    return [
        CannibalizationURLData(
            url=data['url'],
            title=data['title'],
            main_keyword=data['main_keyword'],
            semantic_search_intent=data['semantic_search_intent']
        ) for data in url_data_list
    ]

@app.post("/process_urls_in_batches_for_cannibalization")
async def process_urls_in_batches_for_cannibalization(request: BatchRequest):
    sitemap_url = f"{request.domain.rstrip('/')}/sitemap_index.xml"
    urls = await fetch_sitemap(app.state.client, sitemap_url)

    if not urls:
        raise HTTPException(status_code=404, detail="Sitemap not found or empty")
    
    urls_to_process = urls[request.start:request.start + request.batch_size]
    tasks = [analyze_url_for_cannibalization(url, app.state.client) for url in urls_to_process]
    results = await asyncio.gather(*tasks)
    
    cannibalization_urls = [
        {
            "url": result['url'],
            "title": result.get('title', "No title provided"),
            "main_keyword": result.get('main_keyword', "Not specified"),
            "semantic_search_intent": result.get('semantic_search_intent', "Not specified")
        } for result in results if result
    ]

    next_start = request.start + len(urls_to_process)
    more_batches = next_start < len(urls)

    return {
        "cannibalization_urls": cannibalization_urls,
        "more_batches": more_batches,
        "next_batch_start": next_start if more_batches else None
    }




##############################################



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="debug")