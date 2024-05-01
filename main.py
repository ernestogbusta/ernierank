# main.py

from analyze_url import analyze_url
from analyze_internal_links import analyze_internal_links, InternalLinkAnalysis, correct_url_format
from analyze_wpo import analyze_wpo
from analyze_cannibalization import analyze_cannibalization
from generate_content import get_cached_data, set_cached_data, generate_content_based_on_seo_data, call_openai_gpt4, generate_seo_content, load_processed_data_from_file, process_new_data
from fastapi import FastAPI, HTTPException, Request, Body
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
from aiocache import Cache

# Configuración del logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("CannibalizationAnalysis")

app = FastAPI(title="ErnieRank API")

class BatchRequest(BaseModel):
    domain: str
    batch_size: int = 100  # valor por defecto
    start: int = 0        # valor por defecto para iniciar, asegura que siempre tenga un valor

@app.on_event("startup")
async def startup_event():
    # Configura el logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Configura el cliente HTTP para operaciones asincrónicas
    app.state.client = httpx.AsyncClient()

    # Configura la conexión a Redis con soporte para TLS y autenticación
    redis_url = "rediss://red-co9d0e5jm4es73atc0ng:FuTgObFfNdlhcUdSthCpcXImJIqHtzuq@oregon-redis.render.com:6379"
    app.state.cache = Cache.from_url(redis_url)

    # Verificando la conexión con Redis...
    try:
        # Prueba estableciendo una clave
        await app.state.cache.set('test_key', 'test_value')
        # Prueba obteniendo la clave
        value = await app.state.cache.get('test_key')
        logging.info(f"Valor recuperado de Redis: {value}")
    except Exception as e:
        logging.error(f"Error al conectar con Redis: {e}")

    logging.info("Startup configuration completed.")

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.client.aclose()
    await app.state.cache.close()


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
async def generate_content_endpoint(request: ContentRequest):
    logging.debug(f"Request received for generating content for URL: {request.url}")
    try:
        cached_data = await get_cached_data(request.url)
        if cached_data:
            logging.debug("Using cached data for content generation.")
            if isinstance(cached_data, dict):
                content_generated = generate_seo_content(cached_data)
                return {"generated_content": content_generated}
            else:
                logging.error("Cached data is not in dictionary format.")
                raise TypeError("Cached data must be a dictionary")
        else:
            logging.debug("No cached data found, processing new data.")
            new_data = await process_new_data(request.url, app.state.client)
            if new_data:
                await set_cached_data(request.url, new_data)
                content_generated = generate_seo_content(new_data)
                return {"generated_content": content_generated}
            else:
                raise HTTPException(status_code=500, detail="Failed to process new data")
    except Exception as e:
        logging.error(f"An error occurred while generating content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


##############################################


async def test_redis_connection():
    try:
        # Prueba estableciendo una clave
        await cache.set('test_key', 'test_value')
        # Prueba obteniendo la clave
        value = await cache.get('test_key')
        print(f"Valor recuperado de Redis: {value}")
    except Exception as e:
        print(f"Error al conectar con Redis: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="debug")