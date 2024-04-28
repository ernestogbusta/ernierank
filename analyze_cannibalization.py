from fastapi import HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import asyncio

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CannibalizationAnalysis")

# Definición del modelo Pydantic para los datos de URL
class CannibalizationURLData(BaseModel):
    url: HttpUrl
    title: str

class CannibalizationResult(BaseModel):
    url1: HttpUrl
    url2: HttpUrl
    cannibalization_level: str

# Inicialización de TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')

def clean_text(text: str) -> str:
    """Limpia el texto eliminando caracteres no alfanuméricos y convirtiéndolos a minúsculas."""
    return re.sub(r'\W+', ' ', text).lower()

def should_analyze(url1: HttpUrl, url2: HttpUrl) -> bool:
    """Determina si dos URLs deben ser analizadas para canibalización basado en sus slugs."""
    if url1 == url2 or (url1.rstrip('/') == url2.rstrip('/')):
        return False
    slug1 = url1.strip('/').split('/')[-1]
    slug2 = url2.strip('/').split('/')[-1]
    return slug1 != slug2 and (slug1.startswith(slug2) or slug2.startswith(slug1))

async def calculate_similarity(matrix1, matrix2) -> float:
    """Calcula la similitud del coseno entre dos matrices de términos TF-IDF."""
    return cosine_similarity(matrix1, matrix2)[0][0]

async def analyze_cannibalization(processed_urls: List[CannibalizationURLData]):
    """Analiza la canibalización entre URLs usando la similitud del coseno en los títulos."""
    if not processed_urls:
        logger.error("No URL data provided for cannibalization analysis.")
        raise HTTPException(status_code=400, detail="No URL data provided")

    texts = [clean_text(url.title) for url in processed_urls]
    vectorizer.fit(texts)
    transformed_matrices = [vectorizer.transform([text]) for text in texts]

    results = []
    for i in range(len(processed_urls)):
        for j in range(i + 1, len(processed_urls)):
            if should_analyze(processed_urls[i].url, processed_urls[j].url):
                sim = await calculate_similarity(transformed_matrices[i], transformed_matrices[j])
                if sim > 0.9:
                    level = "High"
                elif sim > 0.6:
                    level = "Medium"
                elif sim > 0.4:
                    level = "Low"
                else:
                    continue
                results.append(CannibalizationResult(
                    url1=processed_urls[i].url,
                    url2=processed_urls[j].url,
                    cannibalization_level=level
                ))

    if results:
        logger.info(f"Cannibalization analysis completed with results: {results}")
    else:
        logger.info("No cannibalization detected")

    return results if results else {"message": "No cannibalization detected"}

