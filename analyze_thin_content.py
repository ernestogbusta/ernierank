import re
import urllib.parse
import asyncio
from fastapi import HTTPException, Response
from pydantic import BaseModel, HttpUrl, validator
from typing import List, Optional, Tuple, Dict, Any
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

async def fetch_processed_data_or_process_batches(domain: str) -> ThinContentRequest:
    logging.debug(f"Iniciando la obtención de datos procesados para el dominio: {domain}")
    processed_data = ThinContentRequest(processed_urls=[
        PageData(
            url='http://example.com/page1',
            title='Example Short Title',
            meta_description='Description is too short.',
            h1='Example H1 Heading',
            h2=['Example H2 Heading', 'Another H2 Heading'],
            main_keyword='example',
            secondary_keywords=['example2', 'example3'],
            semantic_search_intent='example intent'
        ),
        PageData(
            url='http://example.com/page2',
            title='Second Example Title',
            meta_description='Another short description.',
            h1='Second H1 Heading',
            h2=['Second Example H2 Heading'],
            main_keyword='second example',
            secondary_keywords=['second example2', 'second example3'],
            semantic_search_intent='second intent'
        )
    ])
    logging.debug(f"Datos procesados obtenidos para {domain}: {processed_data}")
    return processed_data

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

def analyze_thin_content(processed_urls: List[Dict[str, Any]]):
    thin_content_urls = []

    for url_data in processed_urls:
        url = url_data.get("url")
        title = url_data.get("title")
        meta_description = url_data.get("meta_description")
        main_keyword = url_data.get("main_keyword")
        secondary_keywords = url_data.get("secondary_keywords", [])
        content_length = len(url_data.get("content", ""))  # Assumes content is part of the data

        # Define criteria for thin content
        if content_length < 500 or not title or not meta_description or not main_keyword:
            thin_content_urls.append({
                "url": url,
                "level": "thin content",
                "details": f"Content length: {content_length}, Title: {title}, Meta description: {meta_description}, Main keyword: {main_keyword}"
            })

    return {
        "thin_content_urls": thin_content_urls,
        "total_analyzed": len(processed_urls),
        "thin_content_count": len(thin_content_urls)
    }