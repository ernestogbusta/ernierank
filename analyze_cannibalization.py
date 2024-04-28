from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List
import logging

logging.basicConfig(level=logging.INFO)  # Cambiar a DEBUG si se necesita más detalle en desarrollo
logger = logging.getLogger("CannibalizationAnalysis")

class URLData(BaseModel):
    url: str
    title: str

vectorizer = TfidfVectorizer(stop_words='english')  # Inicializar el vectorizador una vez

def clean_text(text: str) -> str:
    """ Función para limpiar el texto, elimina caracteres no alfanuméricos y pasa a minúsculas. """
    return re.sub(r'\W+', ' ', text).lower()

def should_analyze(url1: str, url2: str) -> bool:
    """ Determina si dos URLs deben ser comparadas para canibalización basado en su estructura y contenido. """
    # Extraer segmentos de URL que podrían indicar eventos específicos o versiones duplicadas
    if url1 == url2 or (url1.rstrip('/') == url2.rstrip('/')):
        return False  # La misma URL o misma base sin importar el final "/"
    slug1 = url1.strip('/').split('/')[-1]
    slug2 = url2.strip('/').split('/')[-1]
    # Verificar si son slugs diferentes completamente y no son variantes directas de la misma página
    if slug1 != slug2 and not (slug1.startswith(slug2) or slug2.startswith(slug1)):
        return False
    return True

async def calculate_similarity(matrix1, matrix2) -> float:
    """ Calcula la similitud del coseno entre dos matrices pre-transformadas. """
    return cosine_similarity(matrix1, matrix2)[0][0]

async def analyze_cannibalization(processed_urls: List[URLData]):
    """ Analiza la canibalización entre URLs dadas usando la similitud del coseno en los títulos. """
    if not processed_urls:
        logger.error("No URL data provided for cannibalization analysis.")
        raise HTTPException(status_code=400, detail="No URL data provided")

    texts = [clean_text(url.title) for url in processed_urls]
    vectorizer.fit(texts)  # Ajustar el vectorizador una vez con todos los textos
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

# Asegúrate de que esta función sea llamada desde un endpoint de FastAPI adecuado.
