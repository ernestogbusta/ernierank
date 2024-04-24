from fastapi import FastAPI, HTTPException, Request, Depends, Query, APIRouter, Body
from fastapi.responses import JSONResponse
import httpx
from httpx import Timeout, AsyncClient
from bs4 import BeautifulSoup
import xmltodict
import os
import json
from pydantic import BaseModel, Field
import uvicorn
from collections import Counter
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urljoin
import re
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from redis.asyncio import Redis
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import subprocess

load_dotenv()
app = FastAPI(title="ErnieRank API")

@app.get("/")
async def root():
    logging.debug("Accediendo a la ruta raíz")
    return {"message": "Hello World"}

@app.on_event("startup")
async def startup_event():
    # Establecer conexión con Redis
    app.state.redis = await Redis(decode_responses=True)

@app.on_event("shutdown")
async def shutdown_event():
    # Cerrar conexión con Redis
    await app.state.redis.close()

@app.get("/some-route")
async def some_route():
    # Usar Redis para cachear o realizar alguna operación
    value = await app.state.redis.get("some_key")
    return {"value": value}

class BatchRequest(BaseModel):
    domain: str
    batch_size: int = 100  # valor por defecto
    start: int = 0        # valor por defecto para iniciar, asegura que siempre tenga un valor

@app.post("/process_urls_in_batches")
async def process_urls_in_batches(request: BatchRequest):
    sitemap_url = f"{request.domain.rstrip('/')}/sitemap_index.xml"
    print(f"Fetching URLs from: {sitemap_url}")
    
    try:
        urls = await fetch_sitemap(app.state.client, sitemap_url, app.state.redis)
    except Exception as e:
        print(f"Failed to fetch sitemap: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch sitemap")

    if not urls:
        print("No URLs found in the sitemap.")
        raise HTTPException(status_code=404, detail="Sitemap not found or empty")
    
    print(f"Total URLs fetched for processing: {len(urls)}")
    urls_to_process = urls[request.start:request.start + request.batch_size]
    print(f"URLs to process from index {request.start} to {request.start + request.batch_size}: {urls_to_process}")

    tasks = [analyze_url(url, app.state.client, app.state.redis) for url in urls_to_process]
    results = await asyncio.gather(*tasks)
    print(f"Results received: {results}")

    valid_results = [{
        "url": result['url'],
        "title": result.get('title', "No title provided"),
        "main_keyword": result.get('main_keyword', "Keyword not specified"),
        "secondary_keywords": result.get('secondary_keywords', []),
        "semantic_search_intent": result.get('semantic_search_intent', "Intent not specified")
    } for result in results if result]

    print(f"Filtered results: {valid_results}")

    next_start = request.start + len(urls_to_process)
    more_batches = next_start < len(urls)
    print(f"More batches: {more_batches}, Next batch start index: {next_start}")

    return {
        "processed_urls": valid_results,
        "more_batches": more_batches,
        "next_batch_start": next_start if more_batches else None
    }

async def fetch_sitemap(client, url, redis_client: Redis, timeout_config=Timeout(10.0, read=60.0)):
    redis_key = f"sitemap:{url}"
    cached_sitemap = await redis_client.get(redis_key)
    if cached_sitemap:
        print(f"Cache hit for sitemap at {url}")
        cached_data = json.loads(cached_sitemap)
        if not cached_data:
            print("Cached sitemap data is empty, refetching...")
        else:
            print(f"Using cached data for {url}")
            return cached_data  # Utilizar los datos del caché si están presentes y son válidos

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        "Accept": "application/xml, application/xhtml+xml, text/html, application/json; q=0.9, */*; q=0.8"
    }

    try:
        response = await client.get(url, headers=headers, timeout=timeout_config)
        if response.status_code != 200:
            print(f"HTTP status code {response.status_code} received for {url}, retrying...")
            response = await client.get(url, headers=headers, timeout=timeout_config)  # Retry once if non-200 received
            if response.status_code != 200:
                print(f"Failed to fetch sitemap after retry for {url}: {response.status_code}")
                return None

        sitemap_contents = xmltodict.parse(response.content)
        all_urls = []

        if 'sitemapindex' in sitemap_contents:
            sitemap_indices = sitemap_contents.get('sitemapindex', {}).get('sitemap', [])
            sitemap_indices = sitemap_indices if isinstance(sitemap_indices, list) else [sitemap_indices]
            child_tasks = [asyncio.create_task(fetch_sitemap(client, sitemap.get('loc'), redis_client, timeout_config)) for sitemap in sitemap_indices]
            children_urls = await asyncio.gather(*child_tasks)
            for urls in children_urls:
                if urls:
                    all_urls.extend(urls)
        elif 'urlset' in sitemap_contents:
            urls = [url.get('loc') for url in sitemap_contents.get('urlset', {}).get('url', [])]
            all_urls.extend(urls)  # Directly add since they are final URLs

        validated_urls = await validate_urls(client, all_urls)
        if validated_urls:
            await redis_client.set(redis_key, json.dumps(validated_urls), ex=86400)  # 24-hour cache expiration
            print(f"Fetched and validated {len(validated_urls)} URLs from the sitemap at {url}.")
            return validated_urls
        else:
            print(f"No valid URLs found in the sitemap at {url}. Not caching empty result.")
            return None
    except Exception as e:
        print(f"Exception occurred while fetching sitemap from {url}: {e}")
        return None

async def fetch_url(client, url, timeout_config, max_redirects=5):
    current_url = url
    tried_urls = set()  # Mantener un registro de las URLs probadas para detectar bucles

    for _ in range(max_redirects):
        if current_url in tried_urls:
            print(f"Detected redirect loop at {current_url}")
            return None
        tried_urls.add(current_url)

        try:
            response = await client.get(current_url, timeout=timeout_config)
            if response.status_code == 200:
                return current_url
            elif response.status_code in (301, 302):
                location = response.headers.get('Location')
                if not location:
                    print(f"Redirect from {current_url} lacks 'Location' header; cannot continue.")
                    return None
                current_url = location  # Actualizar con la nueva URL de redirección
            else:
                print(f"Unexpected status code {response.status_code} at {current_url}")
                return None
        except Exception as e:
            print(f"Exception during fetch of {current_url}: {e}")
            return None

    print(f"Failed to fetch URL after {max_redirects} redirects: {current_url}")
    return None

@app.get("/test-redirect")
async def test_redirect():
    client = app.state.client  # Utiliza el cliente HTTP almacenado en el estado de la app
    url = 'http://example.com/some-redirect-url'
    timeout_config = Timeout(10.0, read=60.0)  # Configura los tiempos de espera apropiadamente

    try:
        final_url = await fetch_url(client, url, timeout_config)
        if final_url:
            # Aquí podrías llamar a otra función para procesar la URL final
            return {"final_url": final_url, "message": "URL fetched successfully"}
        else:
            return {"error": "Failed to fetch the final URL"}
    except Exception as e:
        return {"error": str(e)}

async def validate_urls(client, urls):
    validated_urls = []
    for url in urls:
        status = await check_url_status(client, url)
        if status == 200:
            validated_urls.append(url)
        else:
            print(f"Skipping URL due to non-200 status ({status}): {url}")
    return validated_urls

async def check_url_status(client, url):
    try:
        response = await client.head(url)
        return response.status_code
    except Exception as e:
        print(f"Error checking status for URL {url}: {e}")
        return None

async def validate_urls(client, urls):
    validated_urls = []
    for url in urls:
        status = await check_url_status(client, url)
        if status == 200:
            validated_urls.append(url)
        else:
            print(f"Skipping URL due to non-200 status ({status}): {url}")
    return validated_urls

async def check_url_status(client, url):
    try:
        response = await client.head(url)
        return response.status_code
    except Exception as e:
        print(f"Error checking status for URL {url}: {e}")
        return None

# Define tus stopwords y términos genéricos
stopwords = {"y", "de", "la", "el", "en", "un", "una", "que", "es", "por", "con", "los", "las", "del", "al", "como", "para", "más", "pero", "su", "le", "lo", "se", "a", "o", "e", "nos", "sin", "sobre", "entre", "si", "tu", "mi", "te", "se", "qué", "cómo", "cuándo", "dónde", "por qué", "cuál", "su"}
generic_keywords = {"marketing digital", "community manager", "aula cm", "qué es", "para qué", "para qué sirve", "cómo funciona"}
common_verbs = {"aprender", "crear", "hacer", "usar", "instalar", "optimizar", "mejorar", "configurar", "desarrollar", "construir"}
adjectives = {"mejor", "nuevo", "grande", "importante", "primero", "último"}
question_starts = {"cómo", "qué", "para qué", "cuál es", "dónde"}
verb_synonyms = {
    "aprender": ["estudiar", "entender", "comprender", "capacitarse"],
    "usar": ["utilizar", "emplear", "manejar", "aplicar"],
    "mejorar": ["optimizar", "perfeccionar", "avanzar", "incrementar"],
    "crear": ["desarrollar", "generar", "producir", "construir"],
    "hacer": ["realizar", "ejecutar", "llevar a cabo", "confeccionar"],
    "instalar": ["configurar", "establecer", "implementar"],
    "optimizar": ["mejorar", "maximizar", "refinar"],
    "configurar": ["ajustar", "establecer", "programar"],
    "desarrollar": ["elaborar", "expandir", "crecer"],
    "construir": ["edificar", "fabricar", "montar"],
    "analizar": ["examinar", "evaluar", "revisar"],
    "promover": ["impulsar", "fomentar", "propagar"],
    "integrar": ["unificar", "incorporar", "combinar"],
    "lanzar": ["iniciar", "presentar", "introducir"],
    "gestionar": ["administrar", "manejar", "dirigir"],
    "innovar": ["renovar", "inventar", "pionear"],
    "optimizar": ["enhance", "boost", "elevate"],
    "navegar": ["explorar", "buscar", "revisar"],
    "convertir": ["transformar", "cambiar", "modificar"],
    "monetizar": ["rentabilizar", "capitalizar", "ingresar"],
    "segmentar": ["dividir", "clasificar", "categorizar"],
    "targetear": ["enfocar", "dirigir", "apuntar"],
    "publicar": ["difundir", "emitir", "divulgar"],
    "analizar": ["investigar", "sondear", "escrutar"],
    "comunicar": ["informar", "notificar", "transmitir"],
    "diseñar": ["esbozar", "planificar", "trazar"],
    "innovar": ["actualizar", "modernizar", "revolucionar"],
    "negociar": ["acordar", "tratar", "concertar"],
    "organizar": ["ordenar", "estructurar", "planear"],
    "planificar": ["programar", "proyectar", "prever"],
    "producir": ["fabricar", "generar", "crear"],
    "programar": ["codificar", "desarrollar", "planificar"],
    "promocionar": ["publicitar", "anunciar", "divulgar"],
    "recomendar": ["aconsejar", "sugerir", "proponer"],
    "reducir": ["disminuir", "decrementar", "minimizar"],
    "reforzar": ["fortalecer", "intensificar", "consolidar"],
    "registrar": ["anotar", "inscribir", "documentar"],
    "relacionar": ["vincular", "asociar", "conectar"],
    "remodelar": ["renovar", "reformar", "modernizar"],
    "rentabilizar": ["lucrar", "beneficiar", "capitalizar"],
    "replicar": ["imitar", "reproducir", "copiar"],
    "resolver": ["solucionar", "arreglar", "aclarar"],
    "responder": ["contestar", "reaccionar", "replicar"],
    "restaurar": ["reparar", "recuperar", "renovar"],
    "resultar": ["concluir", "derivarse", "provenir"],
    "retener": ["conservar", "mantener", "preservar"],
    "revelar": ["descubrir", "mostrar", "desvelar"],
    "revisar": ["examinar", "inspeccionar", "corregir"],
    "satisfacer": ["complacer", "contentar", "cumplir"],
    "segmentar": ["dividir", "separar", "clasificar"],
    "seleccionar": ["elegir", "escoger", "preferir"],
    "simplificar": ["facilitar", "clarificar", "depurar"],
    "sintetizar": ["resumir", "condensar", "abreviar"],
    "sistematizar": ["organizar", "ordenar", "estructurar"],
    "solucionar": ["resolver", "remediar", "arreglar"],
    "subrayar": ["enfatizar", "destacar", "resaltar"],
    "sugerir": ["proponer", "indicar", "recomendar"],
    "supervisar": ["controlar", "inspeccionar", "vigilar"],
    "sustituir": ["reemplazar", "cambiar", "suplantar"],
    "transformar": ["cambiar", "modificar", "alterar"],
    "transmitir": ["comunicar", "difundir", "emitir"],
    "valorar": ["evaluar", "apreciar", "estimar"],
    "variar": ["modificar", "alterar", "cambiar"],
    "vender": ["comercializar", "negociar", "ofertar"],
    "verificar": ["comprobar", "confirmar", "validar"],
    "viajar": ["trasladarse", "desplazarse", "moverse"],
    "vincular": ["enlazar", "asociar", "conectar"],
    "visitar": ["recorrer", "frecuentar", "acudir"],
    "visualizar": ["imaginar", "ver", "prever"]
}

def refine_keywords(text: str) -> List[str]:
    words = re.findall(r'\w+', text.lower())
    filtered_words = [word for word in words if word not in stopwords and word not in generic_keywords]
    keywords = []
    for word in filtered_words:
        found = False
        for verb, syn_list in verb_synonyms.items():
            if word in syn_list:
                keywords.append(verb)
                found = True
                break
        if not found:
            keywords.append(word)
    return list(set(keywords))

def calculate_keyword_density(text: str, keywords: List[str]) -> float:
    if not text:
        return 0.0
    words = re.findall(r'\w+', text.lower())
    word_count = Counter(words)
    total_words = sum(word_count.values())
    keyword_hits = sum(word_count[word] for word in keywords if word in word_count)
    if total_words == 0:
        return 0.0
    return (keyword_hits / total_words) * 100

def extract_slug(url: str) -> str:
    parsed_url = urlparse(url)
    path_segments = parsed_url.path.strip("/").split('/')
    return ' '.join(path_segments[-1].replace('-', ' ').split())

def find_keywords(title: str, h1s: List[str], h2s: List[str], text_relevant: str, exclude: List[str]) -> List[str]:
    """
    Identifies and returns the top secondary keywords from the provided content,
    excluding any specified words and prioritizing by relevance and length,
    ensuring they do not contain stop words or digits and are similar to the main keyword.
    """
    content = ' '.join([title] + h1s + h2s + [text_relevant])
    all_keywords = extract_keywords(content)
    # Remove any excluded keywords (typically the main keyword)
    filtered_keywords = [k for k in all_keywords if k not in exclude and not any(word in stopwords or word.isdigit() for word in k.split())]
    # Ensure keywords are similar to the main keyword or repeat the main if no good matches
    final_keywords = [k for k in filtered_keywords if is_similar(k, exclude[0])]
    if not final_keywords:
        final_keywords = [exclude[0]] * 2  # Use main keyword if no suitable secondary keywords
    return final_keywords[:2]  # Return top 2 secondary keywords based on the logic

def is_similar(keyword: str, main_keyword: str) -> bool:
    """
    Check if the keyword is similar to the main keyword. Here, you might implement
    more complex similarity checks based on your specific needs.
    """
    main_parts = set(main_keyword.split())
    keyword_parts = set(keyword.split())
    return len(main_parts & keyword_parts) > 0

def extract_keywords(text: str) -> List[str]:
    """
    Extracts keywords from the given text using n-gram analysis, considering only 2-6 word combinations.
    Filters out any combinations containing stop words or numbers.
    """
    words = re.findall(r'\w+', text.lower())
    n_grams = [' '.join(words[i:i+n]) for n in range(2, 7) for i in range(len(words)-n+1)]
    n_grams = [n_gram for n_gram in n_grams if not any(word in stopwords or word.isdigit() for word in n_gram.split())]
    return n_grams

def filter_and_weight_keywords(keywords: List[str]) -> List[str]:
    """
    Filters keywords by their length and applies weighting logic to prefer keywords
    of 2-3 words over 4-6 words.
    """
    weighted_keywords = {}
    for keyword in keywords:
        words = keyword.split()
        length = len(words)
        if length in (2, 3):
            weight = 5
        elif length == 4:
            weight = 4
        elif length in (5, 6):
            weight = 3
        else:
            continue  # Skip any keywords outside the 2-6 word range

        weighted_keywords[keyword] = weight

    # Sort keywords by weight and then alphabetically within the same weight
    sorted_keywords = sorted(weighted_keywords.items(), key=lambda x: (-x[1], x[0]))
    return [kw for kw, weight in sorted_keywords]

def find_meta_description(soup: BeautifulSoup) -> str:
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        return meta_tag["content"]
    return "No description available"

async def analyze_url(url: str, client: httpx.AsyncClient, redis_client: Redis) -> dict:
    # Saltar URL si es una imagen
    if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp')):
        print(f"Skipping image URL: {url}")
        return None

    # Revisar estado de la URL antes de procesar para asegurar que es una página con status 200
    response = await client.head(url)
    if response.status_code != 200:
        print(f"Skipping URL due to non-200 status: {url} with status {response.status_code}")
        return None

    # Usar caché para evitar procesamiento repetido
    redis_key = f"url_analysis:{url}"
    cached_result = await redis_client.get(redis_key)
    if cached_result:
        print(f"Cache hit for URL: {url}")
        return json.loads(cached_result)

    # Realizar petición GET si el URL no está en caché o necesita actualización
    response = await client.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch URL {url} with status {response.status_code}, skipping...")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.title.text if soup.title else "No title"
    meta_description = find_meta_description(soup)
    h1s = [h1.text.strip() for h1 in soup.find_all('h1')]
    h2s = [h2.text.strip() for h2 in soup.find_all('h2', limit=3)]
    text_relevant = ' '.join([p.text.strip() for p in soup.find_all('p', limit=5)])

    # Extraer palabras clave y slug
    slug = extract_slug(url)
    refined_keywords = refine_keywords(f"{title} {' '.join(h1s)} {' '.join(h2s)} {text_relevant}")
    keyword_density = calculate_keyword_density(text_relevant, refined_keywords)

    # Almacenar y devolver resultados
    result = {
        "url": url,
        "status": response.status_code,
        "title": title,
        "meta_description": meta_description,
        "h1s": h1s,
        "h2s": h2s,
        "slug": slug,
        "refined_keywords": refined_keywords,
        "keyword_density": keyword_density
    }

def calculate_semantic_search_intent(main_keyword, secondary_keywords):
    # Combine main and secondary keywords with given weights
    combined_keywords = [main_keyword] + secondary_keywords
    combined_text = ' '.join(combined_keywords)
    words = combined_text.split()
    
    # Eliminate duplicate words while preserving order
    seen = set()
    unique_words = [x for x in words if not (x in seen or seen.add(x))]
    
    # Ensure the length does not exceed 7 words
    return ' '.join(unique_words[:7])


def extract_relevant_text(soup):
    # Encuentra todos los encabezados principales y extrae texto significativo
    texts = []
    for header in soup.find_all(['h1', 'h2']):
        next_node = header.find_next_sibling()
        while next_node and next_node.name not in ['h1', 'h2', 'h3']:
            if next_node.name == 'p':
                texts.append(next_node.text)
            next_node = next_node.find_next_sibling()
    return ' '.join(texts)

@app.post("/process_all_batches")
async def process_all_batches_endpoint(request: Request):
    body = await request.json()
    domain = body['domain']  # Asumiendo que se envía JSON con un dominio
    return await process_all_batches(domain)

async def process_all_batches(domain):
    start = 0
    batch_size = 100
    more_batches = True
    results = []

    while more_batches:
        response = await process_urls_in_batches(BatchRequest(domain=domain, batch_size=batch_size, start=start))
        results.append(response)
        print(f"Processed batch starting at {start}")
        more_batches = response['more_batches']
        start += batch_size

    return {"result": results}

def get_processed_data():
    # Esta función debe devolver los datos ya procesados que incluyen la 'Semantic Search Intent'
    return [
        {
            "url": "https://aulacm.com/",
            "status": 200,
            "title": "Aula CM - Escuela de Marketing Digital",
            "meta_description": "Especialistas en WordPress, Blogs y Redes Sociales.",
            "main_keyword": "marketing digital",
            "secondary_keywords": ["cursos marketing digital", "agencia marketing digital", "escuela marketing"],
            "semantic_search_intent": "Explora cursos de marketing digital en Aula CM"
        }
        # Añade más diccionarios por cada URL procesada
    ]

def fetch_data_from_api():
    url = "https://ernierank-vd20.onrender.com/process_urls"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch data from API")

def format_data_for_presentation(data):
    formatted_output = []
    for item in data['processed_urls']:  # Asumiendo que esto coincide con tu estructura de respuesta JSON
        if item['status'] == 200:  # Asegurando que solo procesas URLs con status 200
            formatted_output.append(
                f"URL: {item['url']}\n"
                f"Title: {item['title']}\n"
                f"Meta Description: {item['meta_description']}\n"
                f"Main Keyword: {item['main_keyword']}\n"
                f"Secondary Keywords: {', '.join(item['secondary_keywords'])}\n"
                f"Semantic Search Intent: {item['semantic_search_intent']}\n"
            )
    return "\n".join(formatted_output)

def test_api():
    try:
        api_response = fetch_data_from_api()
        presentation_output = format_data_for_presentation(api_response)
        print(presentation_output)
    except Exception as e:
        print(str(e))


########## ENLACES INTERNOS ##########

async def get_http_client():
    return AsyncClient(timeout=10.0)

class LinkAnalysis(BaseModel):
    url: str
    anchor_text: str
    seo_quality: Optional[str] = None
    similarity_score: Optional[float] = None

class InternalLinkAnalysis(BaseModel):
    domain: str
    internal_links_data: List[LinkAnalysis]

class LinkAnalysis(BaseModel):
    url: str
    anchor_text: str
    seo_quality: Optional[str] = None
    similarity_score: Optional[float] = None

class InternalLinkAnalysis(BaseModel):
    domain: str
    internal_links_data: List[LinkAnalysis]

@app.post("/analyze_internal_links", response_model=InternalLinkAnalysis)
async def analyze_internal_links(domain: str = Body(..., embed=True)):
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            sitemap_url = f"{domain.rstrip('/')}/sitemap_index.xml"
            urls = await fetch_sitemap_for_internal_links(client, sitemap_url)
            if not urls:
                logging.error(f"No valid URLs found in sitemap: {sitemap_url}")
                raise HTTPException(status_code=404, detail="No valid URLs found in the sitemap.")
            
            internal_links_data = await process_internal_links(client, urls, domain)
            if not internal_links_data:
                logging.error("No internal links were processed or found.")
                raise HTTPException(status_code=404, detail="No internal links were processed or found.")
            
            return InternalLinkAnalysis(domain=domain, internal_links_data=internal_links_data)
    except Exception as e:
        logging.exception("Failed to analyze internal links")
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_sitemap_for_internal_links(client: httpx.AsyncClient, url: str) -> List[str]:
    try:
        response = await client.get(url)
        if response.status_code == 200:
            sitemap_contents = xmltodict.parse(response.content)
            urls = []
            # Procesar índices de sitemaps
            if 'sitemapindex' in sitemap_contents:
                for sitemap in sitemap_contents['sitemapindex']['sitemap']:
                    child_urls = await fetch_sitemap_for_internal_links(client, sitemap['loc'])
                    urls.extend(child_urls)
            # Procesar sets de URLs
            elif 'urlset' in sitemap_contents:
                urls = [url_entry['loc'] for url_entry in sitemap_contents['urlset']['url']
                        if not url_entry['loc'].endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp'))]
            else:
                logging.error(f"No recognizable sitemap structure found at {url}")
            return urls
        else:
            logging.warning(f"Received non-200 status code {response.status_code} for {url}")
            return []
    except Exception as e:
        logging.error(f"Exception occurred while fetching sitemap from {url}: {e}")
        return []

async def process_internal_links(client: httpx.AsyncClient, urls: List[str], domain: str) -> List[LinkAnalysis]:
    internal_links_data = []
    for url in urls:
        page_content = await fetch_page_content(url, client)
        if page_content:
            soup = BeautifulSoup(page_content, 'html.parser')
            links = extract_internal_links(soup, url, domain)
            for link in links:
                # Asegúrate de usar await aquí para la función coroutine
                seo_quality, similarity_score = await evaluate_link_quality_and_similarity(link['anchor_text'], link['url'])
                internal_links_data.append(LinkAnalysis(url=link['url'], anchor_text=link['anchor_text'], seo_quality=seo_quality, similarity_score=similarity_score))
    return internal_links_data

async def evaluate_link_quality_and_similarity(anchor_text: str, target_url: str) -> (str, float):
    """
    Evalúa la calidad de SEO y calcula la similitud de contenido basado en el texto del ancla y el slug de la URL de destino.
    """
    content_keywords = extract_keywords_from_page(target_url)
    anchor_keywords = set(anchor_text.lower().split())
    content_keywords_set = set(content_keywords)
    overlap = anchor_keywords.intersection(content_keywords_set)
    similarity_score = len(overlap) / len(anchor_keywords) if anchor_keywords else 0

    seo_quality = 'Excellent' if similarity_score > 0.5 else 'Good' if similarity_score > 0 else 'Needs improvement'
    return seo_quality, similarity_score

async def fetch_page_content(url: str, client: httpx.AsyncClient) -> Optional[str]:
    try:
        # Intentamos obtener la URL, siguiendo redirecciones automáticamente
        response = await client.get(url, follow_redirects=True)
        if response.status_code == 200:
            print(f"Contenido recuperado exitosamente para {url}.")
            return response.text
        elif response.status_code in [301, 302]:  # Manejo manual de redirecciones si es necesario
            # Si el cliente no siguió alguna redirección, seguimos manualmente
            new_url = response.headers.get('Location')
            if new_url:
                return await fetch_page_content(new_url, client)
            else:
                print(f"Redirección sin una nueva ubicación para {url}.")
        else:
            print(f"Error al recuperar {url}: Código de estado HTTP {response.status_code}")
    except httpx.RequestError as e:
        print(f"Error de conexión al intentar obtener {url}: {e}")
    except Exception as e:
        print(f"Error inesperado al intentar obtener {url}: {e}")
    return None

def extract_internal_links(soup: BeautifulSoup, base_url: str, domain: str) -> List[Dict[str, str]]:
    links = []
    domain_netloc = urlparse(domain).netloc
    for a_tag in soup.find_all('a', href=True):
        link_url = urljoin(base_url, a_tag['href'])
        link_netloc = urlparse(link_url).netloc
        # Verifica que el enlace sea interno y no apunte a secciones inútiles como anclas o javascript
        if link_netloc == domain_netloc and not urlparse(link_url).path.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
            links.append({
                'url': link_url,
                'anchor_text': a_tag.get_text(strip=True) or 'No Text'
            })
    print(f"Enlaces internos extraídos de {base_url}: {len(links)}")
    return links

async def analyze_url_for_internal_links(client: httpx.AsyncClient, url: str, domain: str) -> Optional[List[LinkAnalysis]]:
    try:
        response = await client.get(url, follow_redirects=True)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            links = extract_internal_links(soup, url, domain)
            if links:
                detailed_links = await evaluate_internal_links(links, url, client)
                return detailed_links
            else:
                print(f"No se encontraron enlaces internos en {url}.")
        else:
            print(f"Error al obtener {url}, código de estado HTTP: {response.status_code}")
    except httpx.RequestError as e:
        print(f"HTTP request failed for {url}: {e}")
    except Exception as e:
        print(f"Unexpected error occurred for {url}: {e}")
    return None

async def evaluate_internal_links(links: List[Dict[str, str]], source_url: str, client: httpx.AsyncClient) -> List[LinkAnalysis]:
    results = []
    source_soup = await fetch_soup(source_url, client)
    if not source_soup:
        return []

    for link in links:
        target_soup = await fetch_soup(link['url'], client)
        if not target_soup:
            results.append(LinkAnalysis(url=link['url'], anchor_text=link['anchor_text'], seo_quality='Target page could not be fetched', similarity_score=0))
        else:
            seo_quality, similarity_score = await evaluate_link_quality_and_similarity(link['anchor_text'], link['url'])
            results.append(LinkAnalysis(url=link['url'], anchor_text=link['anchor_text'], seo_quality=seo_quality, similarity_score=similarity_score))
    return results

async def fetch_soup(url: str, client: httpx.AsyncClient) -> Optional[BeautifulSoup]:
    try:
        response = await client.get(url, follow_redirects=True)
        if response.status_code == 200:
            print(f"Página recuperada correctamente para scraping: {url}")
            return BeautifulSoup(response.content, 'html.parser')
        else:
            logging.warning(f"Error HTTP {response.status_code} al intentar recuperar la URL: {url}")
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP status error while fetching URL {url}: {e.response.status_code}")
    except httpx.RequestError as e:
        logging.error(f"Request error while fetching URL {url}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching URL {url}: {e}")
    return None

def extract_keywords_from_page(url: str) -> List[str]:
    """
    Extrae palabras clave a partir del slug de la URL.
    """
    from urllib.parse import urlparse
    path = urlparse(url).path
    slug = path.strip('/').split('/')[-1]  # Toma el último segmento del path como slug
    keywords = slug.replace('-', ' ').split()
    return keywords

def is_internal_link(link: str, base_url: str) -> bool:
    parsed_link = urlparse(link)
    parsed_base = urlparse(base_url)
    
    # Asegura que el esquema de la URL es válido y pertenece al mismo dominio
    if parsed_link.scheme in ['http', 'https'] and parsed_link.netloc == parsed_base.netloc:
        return True
    
    # Verifica también subdominios como parte del mismo sitio web
    domain_link = '.'.join(parsed_link.netloc.split('.')[-2:])
    domain_base = '.'.join(parsed_base.netloc.split('.')[-2:])
    return domain_link == domain_base


######################################


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_api()
    else:
        uvicorn.run(app, host="0.0.0.0", port=10000)