import re
import urllib.parse
import asyncio
from fastapi import HTTPException
from pydantic import BaseModel, HttpUrl, validator
from typing import List, Optional, Tuple
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class PageData(BaseModel):
    url: HttpUrl
    title: str
    meta_description: Optional[str] = None
    h1: Optional[str] = None
    h2: Optional[List[str]] = []
    main_keyword: Optional[str] = None
    secondary_keywords: List[str]
    semantic_search_intent: str

    @validator('h1', 'meta_description', 'main_keyword', pre=True, always=True)
    def ensure_not_empty(cls, v):
        logging.debug(f"Validando si el campo está vacío: {v}")
        if v == "":
            return None
        return v

    @validator('h2', pre=True, always=True)
    def ensure_list(cls, v):
        logging.debug(f"Validando si h2 es una lista: {v}")
        if v is None:
            return []
        return v

class ThinContentRequest(BaseModel):
    processed_urls: List[PageData]
    more_batches: bool = False
    next_batch_start: Optional[int] = None

    @validator('processed_urls', each_item=True)
    def check_urls(cls, v):
        logging.debug(f"Validando URL y título: {v.url}, {v.title}")
        if not v.title or not v.url:
            raise ValueError("URL and title must be provided for each item.")
        return v

# Asumamos que hemos revisado y confirmado que el max_score es adecuado:
max_score = 1.0  # Puedes ajustar este valor según el máximo real derivado de tu análisis de componentes.

def classify_content_level(normalized_score: float) -> str:
    logging.debug(f"Clasificando el nivel de contenido con puntuación normalizada: {normalized_score}")
    if normalized_score >= 0.6:
        logging.debug("Contenido clasificado como 'high'")
        return "high"
    elif normalized_score >= 0.3:
        logging.debug("Contenido clasificado como 'medium'")
        return "medium"
    elif normalized_score > 0.1:
        logging.debug("Contenido clasificado como 'low'")
        return "low"
    logging.debug("Contenido clasificado como 'none'")
    return "none"

# Precompile regular expressions for efficiency
hyphen_space_pattern = re.compile(r'-')
stopwords = set(["de", "la", "el", "en", "y", "a", "los", "un", "como", "una", "por", "para"])

def clean_and_split(text: str) -> str:
    logging.debug(f"Limpieza y división del texto: {text}")
    if text is None:
        return ''
    return ' '.join(word for word in hyphen_space_pattern.sub(' ', text.lower()).split() if word not in stopwords)

def keyword_in_text(keyword: str, text: str) -> bool:
    """
    Comprueba si todas las palabras de la keyword están en el texto, ignorando el orden y la puntuación.
    """
    if keyword is None or text is None:
        return False  # Retorna falso si la keyword o el texto son None
    keyword_words = set(re.sub(r'[^\w\s]', '', keyword.lower()).split())
    text_words = set(re.sub(r'[^\w\s]', '', text.lower()).split())
    return keyword_words.issubset(text_words)

async def calculate_thin_content_score_and_details(page: PageData, max_score: float = 1.0) -> Tuple[float, str]:
    try:
        score = 0
        issues = []
        total_possible_score = 6.35

        title_normalized = clean_and_split(page.title if page.title else "")
        meta_description_normalized = clean_and_split(page.meta_description if page.meta_description else "")
        h1_normalized = clean_and_split(page.h1 if page.h1 else "")
        keyword_normalized = clean_and_split(page.main_keyword)
        slug_normalized = clean_and_split(urllib.parse.urlparse(page.url).path)

        logging.debug(f"Análisis de título: {title_normalized}, URL: {page.url}")
        if not page.title:
            issues.append(f"No hay title en {page.url}")
            score += 1
        elif len(page.title) < 10:
            issues.append(f"Title muy corto en {page.url}")
            score += 0.8
        if not keyword_in_text(page.main_keyword, page.title):
            issues.append(f"Keyword '{page.main_keyword}' no incluida en title en {page.url}")
            score += 1

        logging.debug(f"Análisis de meta descripción: {meta_description_normalized}")
        if not page.meta_description:
            issues.append(f"No hay meta description en {page.url}")
            score += 0.6
        elif len(page.meta_description) < 50:
            issues.append(f"Meta description muy pobre en {page.url}")
            score += 0.5
        if not keyword_in_text(page.main_keyword, page.meta_description):
            issues.append(f"Keyword '{page.main_keyword}' no incluida en meta description en {page.url}")
            score += 0.25

        logging.debug(f"Análisis de H1: {h1_normalized}")
        if not page.h1:
            issues.append(f"No hay H1 en {page.url}")
            score += 1
        elif len(page.h1) < 10:
            issues.append(f"H1 muy corto en {page.url}")
            score += 0.8
        if not keyword_in_text(page.main_keyword, page.h1):
            issues.append(f"Keyword '{page.main_keyword}' no incluida en H1 en {page.url}")
            score += 0.9

        logging.debug(f"Análisis de H2: {page.h2}")
        if not page.h2:
            issues.append(f"No hay H2 en {page.url}")
            score += 0.7
        else:
            h2_issues = 0
            for h2_text in page.h2:
                h2_normalized = clean_and_split(h2_text)
                if len(h2_text) < 10:
                    h2_issues += 0.5
                if not keyword_in_text(page.main_keyword, h2_text):
                    h2_issues += 0.4
            score += min(h2_issues, 0.7)

        logging.debug(f"Análisis de slug: {slug_normalized}")
        if not keyword_in_text(page.main_keyword, urllib.parse.urlparse(page.url).path):
            issues.append(f"El slug no incluye la keyword '{page.main_keyword}' en {page.url}")
            score += 1

        normalized_score = score / total_possible_score if total_possible_score != 0 else 0
        details = ', '.join(issues) if issues else 'Enhorabuena, no hay errores de thin content'
        return normalized_score, details
    except Exception as e:
        logging.error(f"Error al calcular la puntuación y detalles de thin content: {e}")
        return 0.0, "Error al calcular los detalles de thin content"

async def analyze_thin_content(request: ThinContentRequest):
    if not request.processed_urls:
        raise HTTPException(status_code=404, detail="No URL data available for analysis.")

    logging.debug(f"Procesando análisis de contenido delgado para {len(request.processed_urls)} URLs.")
    tasks = [calculate_thin_content_score_and_details(page) for page in request.processed_urls]
    results = await asyncio.gather(*tasks)

    thin_content_pages = [
        {
            "url": page.url,
            "thin_score": result[0] * max_score,
            "level": classify_content_level(result[0]),
            "details": result[1]
        }
        for page, result in zip(request.processed_urls, results) if classify_content_level(result[0]) != "none"
    ]

    return {"thin_content_pages": thin_content_pages} if thin_content_pages else {"message": "No thin content detected"}
