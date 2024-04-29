import cProfile
import pstats
import io
from fastapi import FastAPI, HTTPException, Request, Body, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
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
import asyncio
import time
import requests
from pydantic import BaseModel, HttpUrl
import logging
from analyze_url import analyze_url
from analyze_internal_links import analyze_internal_links, InternalLinkAnalysis, correct_url_format
from analyze_wpo import analyze_wpo
from analyze_cannibalization import analyze_cannibalization, CannibalizationURLData, fetch_sitemap_urls, extract_title_and_url
from typing import List

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

@app.get("/redirect/{url:path}")
async def handle_redirect(url: str):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            response = await client.get(url)
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Not Found")
            return {"URL Final": str(response.url), "Status": response.status_code}
        except httpx.RequestError as e:
            logger.error(f"Request Error: {str(e)} - URL: {url}")
            raise HTTPException(status_code=500, detail=f"Request Error: {str(e)}")

@app.get("/")
async def read_root():
    return {"Hello": "World"}

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

@app.get("/error")
async def error_handler():
    return {"message": "There was an error with your request."}

class URLData(BaseModel):
    url: HttpUrl
    title: str
    h1: Optional[str] = None
    main_keyword: Optional[str]
    secondary_keywords: List[str] = []
    semantic_search_intent: Optional[str]

@app.post("/test")
async def test_logging(data: List[URLData]):
    logger.debug(f"Received data: {data}")
    if not data:
        logger.info("No data provided.")
        return {"message": "No data provided."}
    else:
        logger.info("Processing data...")
        return {"message": "Data processed."}

class BatchRequest(BaseModel):
    domain: str
    batch_size: int = 100
    start: int = 0

@app.post("/process_urls_in_batches")
async def process_urls_in_batches(request: BatchRequest):
    sitemap_url = f"{request.domain.rstrip('/')}/sitemap_index.xml"
    logger.info(f"Fetching URLs from: {sitemap_url}")
    urls = await fetch_sitemap(app.state.client, sitemap_url)

    if not urls:
        logger.error("No URLs found in the sitemap.")
        raise HTTPException(status_code=404, detail="Sitemap not found or empty")

    logger.info(f"Total URLs fetched for processing: {len(urls)}")
    urls_to_process = urls[request.start:request.start + request.batch_size]
    logger.info(f"URLs to process from index {request.start} to {request.start + request.batch_size}: {urls_to_process}")

    tasks = [analyze_url(url, app.state.client) for url in urls_to_process]
    results = await asyncio.gather(*tasks)
    logger.info(f"Results received: {results}")

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
    logger.info(f"Filtered results: {valid_results}")

    next_start = request.start + len(urls_to_process)
    more_batches = next_start < len(urls)
    logger.info(f"More batches: {more_batches}, Next batch start index: {next_start}")

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

        logger.info(f"Fetched {len(all_urls)} URLs from the sitemap at {url}.")
        return all_urls
    except Exception as e:
        logger.error(f"Error fetching or parsing sitemap from {url}: {str(e)}")
        return []

@app.post("/analyze_internal_links", response_model=InternalLinkAnalysis)
async def handle_analyze_internal_links(domain: str = Body(..., embed=True)):
    corrected_domain = correct_url_format(domain)
    async with httpx.AsyncClient() as client:
        result = await analyze_internal_links(corrected_domain, client)
        return result

def check_server_availability(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Server not available for {url}: {str(e)}")
        return False

if check_server_availability('https://example.com'):
    logger.info("Server is available.")
else:
    logger.error("Server is not available.")

class WPORequest(BaseModel):
    url: str

@app.post("/analyze_wpo")
async def analyze_wpo_endpoint(request: WPORequest):
    return await analyze_wpo(request.url)

class CannibalizationURLData(BaseModel):
    url: HttpUrl
    title: str

class CannibalizationRequest(BaseModel):
    processed_urls: List[CannibalizationURLData]

class CannibalizationResult(BaseModel):
    url1: HttpUrl
    url2: HttpUrl
    cannibalization_level: str

@app.post("/analyze_cannibalization/", response_model=List[CannibalizationResult], status_code=status.HTTP_200_OK)
async def analyze_cannibalization_endpoint(data: CannibalizationRequest):
    logger.debug(f"Received data for cannibalization analysis: {data}")
    try:
        results = await analyze_cannibalization(data.processed_urls)
        return results
    except ValidationError as ve:
        logger.error(f"Validation error: {ve.errors()}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=ve.errors())
    except Exception as e:
        logger.error(f"Unexpected error during cannibalization analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="debug")
