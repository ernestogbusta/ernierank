import re
import urllib.parse
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, HttpUrl, validator

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
        if v == "":
            return None
        return v

    @validator('h2', pre=True, always=True)
    def ensure_list(cls, v):
        if v is None:
            return []
        return v

class ThinContentRequest(BaseModel):
    processed_urls: List[PageData]
    more_batches: bool = False
    next_batch_start: Optional[int] = None

    @validator('processed_urls', each_item=True)
    def check_urls(cls, v):
        if not v.title or not v.url:
            raise ValueError("URL and title must be provided for each item.")
        return v

def fetch_processed_data_or_process_batches(domain: str) -> ThinContentRequest:
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
    return processed_data

# Asumamos que hemos revisado y confirmado que el max_score es adecuado:
max_score = 6.35  # Puedes ajustar este valor según el máximo real derivado de tu análisis de componentes.

# Precompile regular expressions for efficiency
hyphen_space_pattern = re.compile(r'-')
stopwords = set(["de", "la", "el", "en", "y", "a", "los", "un", "como", "una", "por", "para"])

def clean_and_split(text: str) -> str:
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

def classify_content_level(normalized_score: float) -> str:
    if normalized_score >= 0.6:
        return "high"
    elif normalized_score >= 0.3:
        return "medium"
    elif normalized_score > 0.1:
        return "low"
    return "none"

def calculate_thin_content_score_and_details(page: PageData, max_score: float = 6.35) -> Tuple[float, str]:
    score = 0
    issues = []

    title_normalized = clean_and_split(page.title if page.title else "")
    meta_description_normalized = clean_and_split(page.meta_description if page.meta_description else "")
    h1_normalized = clean_and_split(page.h1 if page.h1 else "")
    keyword_normalized = clean_and_split(page.main_keyword)
    slug_normalized = clean_and_split(urllib.parse.urlparse(page.url).path)

    if not page.title:
        issues.append(f"No hay title en {page.url}")
        score += 1
    elif len(page.title) < 10:
        issues.append(f"Title muy corto en {page.url}")
        score += 0.8
    if not keyword_in_text(page.main_keyword, page.title):
        issues.append(f"Keyword '{page.main_keyword}' no incluida en title en {page.url}")
        score += 1

    if not page.meta_description:
        issues.append(f"No hay meta description en {page.url}")
        score += 0.6
    elif len(page.meta_description) < 50:
        issues.append(f"Meta description muy pobre en {page.url}")
        score += 0.5
    if not keyword_in_text(page.main_keyword, page.meta_description):
        issues.append(f"Keyword '{page.main_keyword}' no incluida en meta description en {page.url}")
        score += 0.25

    if not page.h1:
        issues.append(f"No hay H1 en {page.url}")
        score += 1
    elif len(page.h1) < 10:
        issues.append(f"H1 muy corto en {page.url}")
        score += 0.8
    if not keyword_in_text(page.main_keyword, page.h1):
        issues.append(f"Keyword '{page.main_keyword}' no incluida en H1 en {page.url}")
        score += 0.9

    if not page.h2:
        issues.append(f"No hay H2 en {page.url}")
        score += 0.7
    else:
        h2_issues = 0
        for h2_text in page.h2:
            if len(h2_text) < 10:
                h2_issues += 0.5
            if not keyword_in_text(page.main_keyword, h2_text):
                h2_issues += 0.4
        score += min(h2_issues, 0.7)

    if not keyword_in_text(page.main_keyword, urllib.parse.urlparse(page.url).path):
        issues.append(f"El slug no incluye la keyword '{page.main_keyword}' en {page.url}")
        score += 1

    normalized_score = score / max_score if max_score != 0 else 0
    details = ', '.join(issues) if issues else 'Enhorabuena, no hay errores de thin content'
    return normalized_score, details

def analyze_thin_content(processed_urls: List[Dict[str, Any]]) -> Dict[str, Any]:
    thin_content_urls = []

    for url_data in processed_urls:
        page = PageData(**url_data)
        normalized_score, details = calculate_thin_content_score_and_details(page)
        level = classify_content_level(normalized_score)
        
        thin_content_urls.append({
            "url": page.url,
            "level": level,
            "details": details
        })

    return {
        "thin_content_urls": thin_content_urls,
        "total_analyzed": len(processed_urls),
        "thin_content_count": len(thin_content_urls)
    }
