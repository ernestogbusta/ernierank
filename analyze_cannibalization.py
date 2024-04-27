import os
import asyncio
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
import hashlib

# Configuración del logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("CannibalizationAnalysis")

class URLData(BaseModel):
    url: str
    title: str
    meta_description: str
    main_keyword: str
    secondary_keywords: List[str]
    semantic_search_intent: str

# Implementación de un simple caché en memoria
cache = {}

def clean_text(text: str) -> str:
    """Normaliza el texto eliminando caracteres especiales y stopwords."""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return ' '.join(word for word in text.split())

async def get_from_cache(key):
    """ Obtiene un valor del caché si no ha expirado. """
    item = cache.get(key)
    if item and asyncio.get_event_loop().time() < item[1]:
        return item[0]
    return None

async def set_to_cache(key, value, duration=3600):
    """ Guarda un valor en el caché con un tiempo de expiración. """
    expire_at = asyncio.get_event_loop().time() + duration
    cache[key] = (value, expire_at)

async def calculate_similarity(text1: str, text2: str) -> float:
    """Calcula la similitud del coseno entre dos textos utilizando caché."""
    hash_key = hashlib.md5(f"{text1}_{text2}".encode()).hexdigest()
    cached_result = await get_from_cache(hash_key)
    if cached_result is not None:
        return float(cached_result)

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0][0]
    
    # Cachear el resultado
    await set_to_cache(hash_key, str(cosine_sim))
    return cosine_sim

async def analyze_cannibalization(processed_urls: List[URLData]):
    """Analiza URLs de manera asincrónica para detectar canibalización."""
    results = []
    n = len(processed_urls)
    for i in range(n):
        for j in range(i + 1, n):
            sim_title = await calculate_similarity(clean_text(processed_urls[i].title), clean_text(processed_urls[j].title))
            sim_main_keyword = await calculate_similarity(clean_text(processed_urls[i].main_keyword), clean_text(processed_urls[j].main_keyword))
            sim_semantic = await calculate_similarity(clean_text(processed_urls[i].semantic_search_intent), clean_text(processed_urls[j].semantic_search_intent))

            if sim_title > 0.7 and sim_main_keyword > 0.7 and sim_semantic > 0.7:
                level = "Alta"
            elif sim_title > 0.6 and sim_main_keyword > 0.5 and sim_semantic > 0.6:
                level = "Media"
            elif sim_title > 0.5 and sim_main_keyword > 0.3 and sim_semantic > 0.5:
                level = "Baja"
            else:
                continue

            results.append({
                "url1": processed_urls[i].url,
                "url2": processed_urls[j].url,
                "cannibalization_level": level
            })

    return {"cannibalization_issues": results} if results else {"message": "No cannibalization detected"}
