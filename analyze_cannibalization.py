from fastapi import HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class CannibalizationData(BaseModel):
    url: HttpUrl
    semantic_search_intent: str

vectorizer = TfidfVectorizer()

def clean_text(text: str) -> str:
    """Limpia el texto eliminando caracteres no alfanuméricos y convirtiéndolos a minúsculas."""
    return re.sub(r'\W+', ' ', text).lower()

def calculate_similarity(text1: str, text2: str) -> float:
    """Calcula la similitud del coseno entre dos textos."""
    vectorizer.fit([text1, text2])
    matrix = vectorizer.transform([text1, text2])
    return cosine_similarity(matrix)[0, 1]

def slug_similarity(url1: HttpUrl, url2: HttpUrl) -> float:
    """Calcula una medida de similitud basada en los slugs de las URLs."""
    slug1 = url1.strip('/').split('/')[-1].rstrip('-0123456789')
    slug2 = url2.strip('/').split('/')[-1].rstrip('-0123456789')
    if slug1 == slug2:
        return 1.0
    return 0.0  # No similarity for different slugs

def is_likely_edition(slug_part: str) -> bool:
    """Determina si un slug es una edición de evento, considerando números romanos y años."""
    return bool(re.match(r'^([i|v|x|l|c|d|m]+|\d{4})$', slug_part))

def should_analyze(url1: HttpUrl, url2: HttpUrl) -> bool:
    """Determina si dos URLs deben ser analizadas para canibalización basado en sus slugs."""
    slug1 = url1.strip('/').split('/')[-1]
    slug2 = url2.strip('/').split('/')[-1]
    if url1 == url2 or url1.rstrip('/') == url2.rstrip('/'):
        return False
    if is_likely_edition(slug1) or is_likely_edition(slug2):
        return False  # No canibalización si alguno de los slugs es una edición
    return slug1.rstrip('-0123456789') == slug2.rstrip('-0123456789')

async def analyze_cannibalization(processed_urls: List[CannibalizationData]):
    results = []
    for i in range(len(processed_urls)):
        for j in range(i + 1, len(processed_urls)):
            if should_analyze(processed_urls[i].url, processed_urls[j].url):
                semantic_sim = calculate_similarity(clean_text(processed_urls[i].semantic_search_intent), clean_text(processed_urls[j].semantic_search_intent))
                slug_sim = slug_similarity(processed_urls[i].url, processed_urls[j].url)
                overall_sim = (semantic_sim + slug_sim) / 2  # Average similarity

                if overall_sim > 0.9:
                    level = "Alta"
                elif overall_sim > 0.6:
                    level = "Media"
                elif overall_sim > 0.4:
                    level = "Baja"
                else:
                    continue

                results.append({
                    "url1": processed_urls[i].url,
                    "url2": processed_urls[j].url,
                    "cannibalization_level": level
                })

    if results:
        return results
    else:
        raise HTTPException(status_code=404, detail="No cannibalization detected")
