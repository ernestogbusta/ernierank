from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List

# Configuración de logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("CannibalizationAnalysis")

# Modelo de datos
class URLData(BaseModel):
    url: str
    title: str
    meta_description: str
    main_keyword: str
    secondary_keywords: List[str]
    semantic_search_intent: str

# Configuración del vectorizador Tfidf
vectorizer = TfidfVectorizer(stop_words='english')

def clean_text(text: str) -> str:
    """Limpiar el texto para la preparación."""
    cleaned_text = re.sub(r'\W+', ' ', text).lower()
    logger.debug(f"Text cleaned: {cleaned_text}")
    return cleaned_text

async def calculate_similarity(text1: str, text2: str) -> float:
    """Calcular la similitud del coseno entre dos textos."""
    if text1 == text2:
        return 1.0
    matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(matrix[0:1], matrix[1:])[0][0]
    logger.debug(f"Calculated cosine similarity: {cosine_sim} for texts: [{text1}], [{text2}]")
    return cosine_sim

async def analyze_cannibalization(processed_urls: List[URLData]):
    """Analizar la canibalización entre URLs dadas centrada en la intención semántica de búsqueda."""
    if not processed_urls:
        logger.error("No URL data provided for cannibalization analysis.")
        raise HTTPException(status_code=400, detail="No URL data provided")
    
    texts = [clean_text(url.semantic_search_intent) for url in processed_urls]  # Pre-clean texts for efficiency
    results = []
    logger.info(f"Starting cannibalization analysis for {len(processed_urls)} URLs.")
    
    for i in range(len(processed_urls)):
        for j in range(i + 1, len(processed_urls)):
            sim_semantic = await calculate_similarity(texts[i], texts[j])

            if sim_semantic > 1:
                level = "Alta"
            elif sim_semantic > 0.95:
                level = "Media"
            elif sim_semantic > 0.9:
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
