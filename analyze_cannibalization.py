from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List
import logging
import httpx
from bs4 import BeautifulSoup
import xmltodict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CannibalizationAnalysis")

class CannibalizationURLData(BaseModel):
    url: str
    title: str

vectorizer = TfidfVectorizer(stop_words='english')

def clean_text(text: str) -> str:
    return re.sub(r'\W+', ' ', text).lower()

def should_analyze(url1: str, url2: str) -> bool:
    if url1 == url2 or (url1.rstrip('/') == url2.rstrip('/')):
        return False
    slug1 = url1.strip('/').split('/')[-1]
    slug2 = url2.strip('/').split('/')[-1]
    if slug1 != slug2 and not (slug1.startswith(slug2) or slug2.startswith(slug1)):
        return False
    return True

async def calculate_similarity(matrix1, matrix2) -> float:
    return cosine_similarity(matrix1, matrix2)[0][0]

async def analyze_cannibalization(processed_urls: List[CannibalizationURLData]):
    """Analiza la canibalización entre URLs dadas usando la similitud del coseno en los títulos."""
    if not processed_urls:
        logger.error("No URL data provided for cannibalization analysis.")
        raise HTTPException(status_code=400, detail="No URL data provided")

    texts = [clean_text(url.title) for url in processed_urls]
    vectorizer.fit(texts)
    transformed_matrices = [vectorizer.transform([text]) for text in texts]

    results = []
    for i in range(len(processed_urls)):
        for j in range(i + 1, len(processed_urls)):
            if not should_analyze(processed_urls[i].url, processed_urls[j].url):
                continue
            sim = await calculate_similarity(transformed_matrices[i], transformed_matrices[j])
            if sim > 0.9:
                level = "Alta"
            elif sim > 0.6:
                level = "Media"
            elif sim > 0.4:
                level = "Baja"
            else:
                continue
            results.append({
                "url1": processed_urls[i].url,
                "url2": processed_urls[j].url,
                "cannibalization_level": level
            })

    if results:
        logger.info(f"Cannibalization analysis completed with results: {results}")
    else:
        logger.info("No cannibalization detected")

    return {"cannibalization_issues": results} if results else {"message": "No cannibalization detected"}

async def fetch_sitemap_urls(client, sitemap_url):
    try:
        response = await client.get(sitemap_url)
        sitemap_contents = xmltodict.parse(response.content)
        urls = [url['loc'] for url in sitemap_contents['urlset']['url']]
        return urls
    except Exception as e:
        logger.error(f"Error fetching sitemap: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch sitemap")

async def extract_title_and_url(client, url):
    try:
        response = await client.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else ""
            return CannibalizationURLData(url=url, title=title)
        else:
            return None
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        return None
