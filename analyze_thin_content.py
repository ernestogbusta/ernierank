from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import urllib.parse

class PageData(BaseModel):
    url: str
    title: str
    meta_description: str
    h1: Optional[str] = None  # Añadir h1 como opcional si no siempre está presente
    h2: Optional[List[str]] = []  # Añadir h2, puede ser una lista de strings
    main_keyword: str
    secondary_keywords: List[str]
    semantic_search_intent: str

class ThinContentRequest(BaseModel):
    processed_urls: List[PageData]
    more_batches: Optional[bool] = False
    next_batch_start: Optional[int] = None

# Funciones de análisis
async def fetch_processed_data_or_process_batches(domain: str) -> ThinContentRequest:
    # Asumimos que se pueden obtener datos de algún servicio o base de datos
    return ThinContentRequest(processed_urls=[
        # Ejemplo de datos simulados
        PageData(
            url='http://example.com/page1',
            title='Example Short Title',
            meta_description='Description is too short.',
            main_keyword='example',
            secondary_keywords=['example2', 'example3'],
            semantic_search_intent='example intent'
        )
    ])

async def analyze_thin_content(request: ThinContentRequest):
    if not request.processed_urls:
        raise HTTPException(status_code=404, detail="No URL data available for analysis.")

    thin_content_pages = []
    max_score = 3.45  # Suma de todas las ponderaciones ajustadas

    for page in request.processed_urls:
        score, details = calculate_thin_content_score_and_details(page)
        content_level = "low"
        if score / max_score >= 0.75:
            content_level = "high"
        elif score / max_score >= 0.5:
            content_level = "medium"
        if score / max_score >= 0.25:  # Umbral de decisión para contenido 'Thin'
            thin_content_pages.append({"url": page.url, "thin_score": score, "level": content_level, "details": details})

    if not thin_content_pages:
        return {"message": "No thin content detected"}
    else:
        return {"thin_content_pages": thin_content_pages}


def calculate_thin_content_score_and_details(page: PageData, stopwords=["de", "la", "el", "en", "y", "a", "los", "un", "como"]) -> Tuple[float, str]:
    score = 0
    issues = []

    # Helper function to remove stopwords and split into words
    def clean_and_split(text):
        if text is None:
            return []
        # Reemplaza los guiones con espacios para normalizar el slug y el título
        cleaned_text = text.replace('-', ' ').lower()
        return ' '.join(word for word in cleaned_text.split() if word not in stopwords)

    # Normaliza y divide el título, keyword y slug para comparación
    title_normalized = clean_and_split(page.title)
    keyword_normalized = clean_and_split(page.main_keyword)
    slug_normalized = clean_and_split(urllib.parse.urlparse(page.url).path)

    if not page.title or len(page.title) < 10:
        score += 1
        issues.append(f"title muy corto o ausente en {page.url}")

    if keyword_normalized not in title_normalized:
        score += 0.5
        issues.append(f"keyword principal '{page.main_keyword}' no incluido adecuadamente en el título")

    if not page.meta_description or len(page.meta_description) < 50:
        score += 0.25
        issues.append(f"meta description muy corta en {page.url}")

    if hasattr(page, 'h1') and page.h1:
        h1_normalized = clean_and_split(page.h1)
        if len(page.h1) < 10:
            score += 0.9
            issues.append(f"h1 no encontrado o demasiado corto en {page.url}")

        if keyword_normalized not in h1_normalized:
            score += 0.5
            issues.append(f"keyword principal '{page.main_keyword}' no incluido adecuadamente en h1")

    if hasattr(page, 'h2') and page.h2:
        h2_normalized = [clean_and_split(h) for h in page.h2]
        if not page.h2 or all(len(h) < 10 for h in page.h2):
            score += 0.5
            issues.append(f"h2 no encontrado o demasiado corto en {page.url}")

        if all(keyword_normalized not in h for h in h2_normalized):
            score += 0.5
            issues.append(f"keyword principal '{page.main_keyword}' no incluido adecuadamente en h2")

    if keyword_normalized not in slug_normalized:
        score += 0.8
        issues.append(f"slug no contiene adecuadamente el keyword principal '{page.main_keyword}'")

    details = ', '.join(issues) if issues else 'No significant issues detected.'
    return score, details
