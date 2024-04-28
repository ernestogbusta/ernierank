import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import hashlib
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List

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

pattern = re.compile(r'\W+')

def clean_text(text: str) -> str:
    """Normaliza el texto eliminando caracteres especiales y stopwords."""
    cleaned_text = pattern.sub(' ', text).lower()
    logger.debug(f"Text cleaned: {cleaned_text}")
    return cleaned_text

async def get_from_cache(key):
    """Obtiene un valor del caché si no ha expirado."""
    item = cache.get(key)
    if item:
        logger.debug(f"Cache hit for key: {key}")
        return item
    logger.debug(f"Cache miss for key: {key}")
    return None

async def set_to_cache(key, value, duration=3600):
    """Guarda un valor en el caché con un tiempo de expiración."""
    cache[key] = (value, duration)
    logger.debug(f"Value set in cache for key: {key} with expiration: {duration}")

async def calculate_similarity(text1: str, text2: str) -> float:
    """Calcula la similitud del coseno entre dos textos utilizando caché."""
    hash_key = hashlib.md5(f"{text1}_{text2}".encode()).hexdigest()
    cached_result = await get_from_cache(hash_key)
    if cached_result:
        return float(cached_result[0])

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0][0]
    
    await set_to_cache(hash_key, str(cosine_sim))
    logger.debug(f"Calculated cosine similarity: {cosine_sim} for texts: [{text1}], [{text2}]")
    return cosine_sim

async def analyze_cannibalization(processed_urls: List[URLData]):
    """Analiza URLs de manera asincrónica para detectar canibalización."""
    if not processed_urls:
        logger.error("No URL data provided for cannibalization analysis.")
        raise HTTPException(status_code=400, detail="No URL data provided")
    
    results = []
    logger.info(f"Starting cannibalization analysis for {len(processed_urls)} URLs.")
    for i in range(len(processed_urls)):
        for j in range(i + 1, len(processed_urls)):
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
            logger.debug(f"Cannibalization issue detected: {results[-1]}")

    if results:
        logger.info(f"Cannibalization analysis completed with results: {results}")
    else:
        logger.info("No cannibalization detected")
    return {"cannibalization_issues": results} if results else {"message": "No cannibalization detected"}
