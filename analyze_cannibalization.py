from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, ValidationError
from typing import List
import logging
import asyncio
import httpx
from bs4 import BeautifulSoup
import xmltodict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("CannibalizationAnalysis")

app = FastAPI()

class CannibalizationURLData(BaseModel):
    url: HttpUrl
    title: str

vectorizer = TfidfVectorizer(stop_words='english')

def clean_text(text: str) -> str:
    """ Limpiar el texto eliminando caracteres no alfanuméricos y convirtiéndolos a minúsculas. """
    return re.sub(r'\W+', ' ', text).lower()

def should_analyze(url1: HttpUrl, url2: HttpUrl) -> bool:
    """ Determinar si se debe analizar canibalización entre dos URLs. """
    if url1 == url2 or (url1.rstrip('/') == url2.rstrip('/')):
        return False
    slug1 = url1.strip('/').split('/')[-1]
    slug2 = url2.strip('/').split('/')[-1]
    return slug1 != slug2 and (slug1.startswith(slug2) or slug2.startswith(slug1))

async def calculate_similarity(matrix1, matrix2) -> float:
    """ Calcular la similitud del coseno entre dos matrices de términos TF-IDF. """
    return cosine_similarity(matrix1, matrix2)[0][0]

async def analyze_cannibalization(processed_urls: List[CannibalizationURLData]):
    """ Analizar la canibalización entre URLs usando la similitud del coseno en los títulos. """
    if not processed_urls:
        logger.error("No URL data provided for cannibalization analysis.")
        raise HTTPException(status_code=400, detail="No URL data provided")

    try:
        texts = [clean_text(url.title) for url in processed_urls]
        vectorizer.fit(texts)
        transformed_matrices = [vectorizer.transform([text]) for text in texts]

        results = []
        for i in range(len(processed_urls)):
            for j in range(i + 1, len(processed_urls)):
                if should_analyze(processed_urls[i].url, processed_urls[j].url):
                    sim = await calculate_similarity(transformed_matrices[i], transformed_matrices[j])
                    level = "None"
                    if sim > 0.9:
                        level = "Alta"
                    elif sim > 0.6:
                        level = "Media"
                    elif sim > 0.4:
                        level = "Baja"
                    if level != "None":
                        results.append({
                            "url1": processed_urls[i].url,
                            "url2": processed_urls[j].url,
                            "cannibalization_level": level
                        })
                    logger.debug(f"Processed pair: {processed_urls[i].url} and {processed_urls[j].url} with similarity {sim} and level {level}")
        if results:
            logger.info(f"Cannibalization analysis completed with results: {results}")
        else:
            logger.info("No cannibalization detected")
        return results if results else [{"message": "No cannibalization detected"}]
    except Exception as e:
        logger.error(f"Error during cannibalization analysis: {e}")
        raise HTTPException(status_code=500, detail="Error processing cannibalization analysis")

async def fetch_sitemap_urls(client: httpx.AsyncClient, sitemap_url: str):
    """ Obtener URLs desde un sitemap. """
    try:
        response = await client.get(sitemap_url)
        sitemap_contents = xmltodict.parse(response.content)
        urls = [url['loc'] for url in sitemap_contents['urlset']['url']]
        logger.debug(f"URLs fetched from sitemap: {urls}")
        return urls
    except Exception as e:
        logger.error(f"Error fetching sitemap: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch sitemap")

async def extract_title_and_url(client: httpx.AsyncClient, url: str):
    """ Extraer título y URL de una página web. """
    try:
        response = await client.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else ""
            logger.debug(f"Title extracted for URL {url}: {title}")
            return CannibalizationURLData(url=url, title=title)
        else:
            logger.error(f"Failed to fetch valid response for URL {url}: Status code {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        return None