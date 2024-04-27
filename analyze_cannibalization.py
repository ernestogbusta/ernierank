from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging

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

STOPWORDS = set("""
a al algo algunos alguna algunas allí ante antes bajo ambos cada cual cuál cada cuanto cuánto cuanta cuánta cuantas cuántas cuanto cuánto cuantos cuántos con contra cuando cuándo cómo cómo de del desde donde dónde durante él ella ellas ellos ese esa eso esos esas esta estas este estos esta ésta éstas éste éstos etcétera ha hace hacia hasta incluso la las le les lo los más menos mi mí mis mucho muchos nada ni no nos nosotros nuestra nuestras nuestro nuestros os para pero por porque que quién quiénes quiénes qué qué se sea sean sido sobre solo son su sus suya suyas suyo suyos sí tal también tanto te tú tu tus tuve tuviste suyo suyos suya suyas un una uno unos unas usted ustedes vosotros vuestra vuestras vuestro vuestros ya yo""".split())

def clean_text(text: str) -> str:
    """Normaliza el texto eliminando caracteres especiales y stopwords."""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return ' '.join(word for word in text.split() if word not in STOPWORDS)

def calculate_similarity(text1: str, text2: str) -> float:
    """Calcula la similitud del coseno entre dos textos."""
    vectorizer = TfidfVectorizer(stop_words='english')  # Cambiado a 'english' ya que 'spanish' no es soportado directamente
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0][0]
    return cosine_sim

def analyze_cannibalization(processed_urls: List[URLData]):
    """Analiza URLs para detectar canibalización basada en la similitud textual de componentes clave y clasifica el nivel de canibalización."""
    results = []
    n = len(processed_urls)
    for i in range(n):
        for j in range(i + 1, n):
            # Calcula las similitudes
            sim_title = calculate_similarity(clean_text(processed_urls[i].title), clean_text(processed_urls[j].title))
            sim_main_keyword = calculate_similarity(clean_text(processed_urls[i].main_keyword), clean_text(processed_urls[j].main_keyword))
            sim_semantic = calculate_similarity(clean_text(processed_urls[i].semantic_search_intent), clean_text(processed_urls[j].semantic_search_intent))

            # Define los umbrales de similitud para clasificar la canibalización
            if sim_title > 0.7 and sim_main_keyword > 0.7 and sim_semantic > 0.7:
                level = "Alta"
            elif sim_title > 0.6 and sim_main_keyword > 0.5 and sim_semantic > 0.6:
                level = "Media"
            elif sim_title > 0.5 and sim_main_keyword > 0.3 and sim_semantic > 0.5:
                level = "Baja"
            else:
                continue  # No se considera canibalización si no alcanza el mínimo umbral

            # Agrega los resultados a la lista
            results.append({
                "url1": processed_urls[i].url,
                "url2": processed_urls[j].url,
                "title_similarity": sim_title,
                "keyword_similarity": sim_main_keyword,
                "semantic_similarity": sim_semantic,
                "cannibalization_level": level
            })

    # Devuelve los problemas de canibalización detectados o un mensaje si no se detectó ninguno
    return {"cannibalization_issues": results} if results else {"message": "No cannibalization detected"}
