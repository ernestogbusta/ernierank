import re
import urllib.parse
import asyncio
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
import logging

class PageData(BaseModel):
    url: str
    title: str
    meta_description: str
    h1: Optional[str] = None
    h2: Optional[List[str]] = []
    main_keyword: str
    secondary_keywords: List[str]
    semantic_search_intent: str

class ThinContentRequest(BaseModel):
    processed_urls: List[PageData]
    more_batches: Optional[bool] = False
    next_batch_start: Optional[int] = None

async def fetch_processed_data_or_process_batches(domain: str) -> ThinContentRequest:
    logging.debug(f"Iniciando la obtención de datos procesados para el dominio: {domain}")
    # Simulación de datos, estos deberían ser extraídos de tu sistema de gestión de contenidos o base de datos
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
    """
    Classifica el nivel de contenido en función del puntaje de contenido delgado normalizado.
    """
    logging.debug(f"Clasificando el nivel de contenido con puntuación normalizada: {normalized_score}")
    if normalized_score >= 0.6:
        logging.info("Contenido clasificado como 'high'")
        return "high"
    elif normalized_score >= 0.3:
        logging.info("Contenido clasificado como 'medium'")
        return "medium"
    elif normalized_score > 0:
        logging.info("Contenido clasificado como 'low'")
        return "low"
    logging.info("Contenido clasificado como 'none'")
    return "none"

# Precompile regular expressions for efficiency
hyphen_space_pattern = re.compile(r'-')
stopwords = set(["de", "la", "el", "en", "y", "a", "los", "un", "como", "una", "por"])

def clean_and_split(text: str) -> str:
    """Cleans and splits the text by removing specified stopwords and replacing hyphens with spaces."""
    return ' '.join(word for word in hyphen_space_pattern.sub(' ', text.lower()).split() if word not in stopwords)

async def calculate_thin_content_score_and_details(page: PageData, max_score: float = 1.0) -> Tuple[float, str]:
    score = 0
    issues = []
    total_possible_score = 6.35  # Suma máxima de todas las penalizaciones más severas posibles

    title_normalized = clean_and_split(page.title)
    keyword_normalized = clean_and_split(page.main_keyword)
    slug_normalized = clean_and_split(urllib.parse.urlparse(page.url).path)

    # Title Analysis
    if not page.title:
        issues.append(f"No hay título SEO en {page.url}")
        score += 1
    elif len(page.title) < 10:
        issues.append(f"Título SEO muy corto en {page.url}")
        score += 0.8
    if keyword_normalized not in title_normalized:
        issues.append(f"Keyword '{page.main_keyword}' no incluida en el título SEO en {page.url}")
        score += 1

    # Meta Description Analysis
    if not page.meta_description:
        issues.append(f"No meta description in {page.url}")
        score += 0.6
    elif len(page.meta_description) < 50:
        issues.append(f"meta description too short in {page.url}")
        score += 0.5
    if page.meta_description and keyword_normalized not in clean_and_split(page.meta_description):
        issues.append(f"Keyword '{page.main_keyword}' no incluida en la meta description en {page.url}")
        score += 0.25

    # H1 Analysis
    if not page.h1:
        issues.append(f"No hay H1 en {page.url}")
        score += 1
    elif len(page.h1) < 10:
        issues.append(f"h1 muy corto en {page.url}")
        score += 0.8
    if page.h1 and keyword_normalized not in clean_and_split(page.h1):
        issues.append(f"Keyword '{page.main_keyword}' no incluida en el h1 {page.url}")
        score += 0.9

    # H2 Analysis
    if not page.h2:
        issues.append(f"No hay h2 en {page.url}")
        score += 0.7
    else:
        h2_issues = 0
        for h2_text in page.h2:
            h2_normalized = clean_and_split(h2_text)
            if len(h2_text) < 10:
                h2_issues += 0.5
            if keyword_normalized not in h2_normalized:
                h2_issues += 0.4
        score += min(h2_issues, 0.7)  # No exceder 0.7 en total para todos los H2

    # Slug Analysis
    if keyword_normalized not in slug_normalized:
        issues.append(f"El slug no incluye la keyword principal '{page.main_keyword}' at {page.url}")
        score += 1

    normalized_score = score / total_possible_score if total_possible_score != 0 else 0  # Evitar división por cero
    details = ', '.join(issues) if issues else 'No hay errores de thin content'
    return normalized_score, details

async def analyze_thin_content(request: ThinContentRequest):
    if not request.processed_urls:
        raise HTTPException(status_code=404, detail="No URL data available for analysis.")

    tasks = [calculate_thin_content_score_and_details(page) for page in request.processed_urls]
    results = await asyncio.gather(*tasks)

    thin_content_pages = [
        {
            "url": page.url,
            "thin_score": result[0] * max_score,  # Muestra el score en escala real de 0 a 1
            "level": classify_content_level(result[0]),
            "details": result[1]
        }
        for page, result in zip(request.processed_urls, results) if classify_content_level(result[0]) != "none"
    ]

    return {"thin_content_pages": thin_content_pages} if thin_content_pages else {"message": "No thin content detected"}