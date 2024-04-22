from fastapi import FastAPI, HTTPException, Request, Depends, Query
import httpx
from httpx import Timeout
from bs4 import BeautifulSoup
import xmltodict
import os
import json
from pydantic import BaseModel
import uvicorn
from collections import Counter
from typing import List, Dict, Optional
from urllib.parse import urlparse
import re
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from redis.asyncio import Redis
from dotenv import load_dotenv
import subprocess

class BatchRequest(BaseModel):
    domain: str
    batch_size: int = 50  # valor por defecto
    start: int = 0        # valor por defecto para iniciar, asegura que siempre tenga un valor

load_dotenv()
app = FastAPI(title="ErnieRank API")

# Middleware para manejar la conexión de Redis
class RedisMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.redis = app.state.redis
        response = await call_next(request)
        return response

# Eventos de inicio y cierre para configurar y cerrar Redis
@app.on_event("startup")
async def startup_event():
    timeout = Timeout(15.0, read=180.0)  # 15 segundos para conectar, 180 segundos para leer
    app.state.redis = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)
    app.state.client = httpx.AsyncClient(timeout=timeout)

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.redis.close()
    await app.state.client.aclose()  # Properly close the client

app.add_middleware(RedisMiddleware)

@app.get("/")
async def root(request: Request):
    redis_client = request.state.redis
    await redis_client.set("key", "value")
    value = await redis_client.get("key")
    return {"hello": "world", "key": value}

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
    batch_size = 50
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_api()
    else:
        uvicorn.run(app, host="0.0.0.0", port=10000)