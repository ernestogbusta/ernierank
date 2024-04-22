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
from difflib import SequenceMatcher
import logging


load_dotenv()

redis_host = os.getenv("REDIS_URL", "redis://red-co9d0e5jm4es73atc0ng:6379")
redis = Redis.from_url(redis_host)

app = FastAPI(title="ErnieRank API")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BatchRequest(BaseModel):
    domain: str
    batch_size: int = 50  # valor por defecto
    start: int = 0        # valor por defecto para iniciar, asegura que siempre tenga un valor

# Middleware para manejar la conexión de Redis
class RedisMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.redis = app.state.redis
        response = await call_next(request)
        return response

@app.on_event("startup")
async def startup_event():
    # Configuración del cliente HTTP
    app.state.client = httpx.AsyncClient()
    
    # Configura el cliente de Redis para usar la URL interna de Redis
    app.state.redis = Redis.from_url("redis://red-co9d0e5jm4es73atc0ng:6379", decode_responses=True, encoding="utf-8")
    
    # Llamada al endpoint de pre-calentamiento
    try:
        preheat_response = await preheat()
        logging.info(f"Preheat completed successfully: {preheat_response}")
    except Exception as e:
        logging.error(f"Preheat failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Preheat process failed during startup")

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.client.aclose()
    app.state.redis.close()
    logging.info("Application shutdown complete.")

app.add_middleware(RedisMiddleware)

@app.get("/")
async def root(request: Request):
    redis_client = request.state.redis
    await redis_client.set("key", "value")
    value = await redis_client.get("key")
    return {"hello": "world", "key": value}

@app.get("/preheat")
async def preheat():
    logging.info("Starting preheat process.")
    try:
        # Redis connection test
        await app.state.redis.set("test", "value")
        test_value = await app.state.redis.get("test")
        if test_value == "value":
            logging.info("Redis preheat successful.")
        else:
            logging.error("Redis preheat failed.")
            raise Exception("Redis test failed")

        # HTTP client test
        response = await app.state.client.get("https://example.com")
        if response.status_code == 200:
            logging.info("HTTP client preheat successful.")
        else:
            logging.error("HTTP client preheat failed.")
            raise Exception("HTTP client test failed")

        logging.info("Preheat process completed successfully.")
        return {"status": "OK"}
    except Exception as e:
        logging.error(f"Preheat failed: {e}")
        return {"status": "Failed", "reason": str(e)}
    
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

async def fetch_sitemap(client, url, redis_client: Redis, timeout_config=Timeout(5.0, read=150.0)):
    redis_key = f"sitemap:{url}"
    cached_sitemap = await redis_client.get(redis_key)
    if cached_sitemap:
        print(f"Cache hit for sitemap at {url}")
        cached_data = json.loads(cached_sitemap)
        if not cached_data:
            print("Cached sitemap data is empty, refetching...")
        else:
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
            for sitemap in sitemap_indices:
                sitemap_url = sitemap.get('loc')
                child_urls = await fetch_sitemap(client, sitemap_url, redis_client, timeout_config)
                if child_urls:
                    all_urls.extend(child_urls)
        elif 'urlset' in sitemap_contents:
            urls = [url.get('loc') for url in sitemap_contents.get('urlset', {}).get('url', [])]
            all_urls.extend(urls)  # Directly add since they are final URLs

        if all_urls:
            await redis_client.set(redis_key, json.dumps(all_urls), ex=86400)  # 24-hour cache expiration
            print(f"Fetched {len(all_urls)} URLs from the sitemap at {url}.")
        else:
            print(f"No URLs found in the sitemap at {url}. Not caching empty result.")
            return None
    except Exception as e:
        print(f"Exception occurred while fetching sitemap from {url}: {e}")
        return None
    return all_urls

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
    timeout_config = Timeout(5.0, read=30.0)  # Configura los tiempos de espera apropiadamente

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
    redis_key = f"url_analysis:{url}"
    cached_result = await redis_client.get(redis_key)
    if cached_result:
        print(f"Cache hit for URL: {url}")
        return json.loads(cached_result)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    try:
        response = await client.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Skipping URL due to non-200 status code: {url} with status {response.status_code}")
            return None

        soup = BeautifulSoup(response.content, 'html.parser')
        slug = extract_slug(url)
        title = soup.title.text if soup.title else "No title"
        meta_description = find_meta_description(soup)
        h1s = [h1.text.strip() for h1 in soup.find_all('h1')]
        h2s = [h2.text.strip() for h2 in soup.find_all('h2', limit=3)]
        text_relevant = ' '.join([p.text.strip() for p in soup.find_all('p', limit=5)])

        main_keyword = slug
        secondary_keywords = find_keywords(title, h1s, h2s, text_relevant, exclude=[slug])
        semantic_search_intent = calculate_semantic_search_intent(main_keyword, secondary_keywords)

        result = {
            "url": url,
            "title": title,
            "meta_description": meta_description,
            "main_keyword": main_keyword,
            "secondary_keywords": secondary_keywords,
            "semantic_search_intent": semantic_search_intent
        }

        await redis_client.set(redis_key, json.dumps(result), ex=86400)
        return result
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None

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



#################### ENLACES INTERNOS

@app.post("/internal_links_analysis")
async def internal_links_analysis(request: Request):
    body = await request.json()
    domain = body['domain']  # Asumiendo que se envía JSON con un dominio
    return await analyze_internal_links(domain)

async def analyze_internal_links(domain: str):
    processed_data = await get_processed_data_from_storage(domain, app.state.redis)

    internal_links = []
    for item in processed_data:
        internal_links.extend(find_internal_links(item))

    seo_recommendations = calculate_seo_recommendations(internal_links)

    return {"internal_links": internal_links, "seo_recommendations": seo_recommendations}

def find_internal_links(data):
    """
    Encuentra enlaces internos en el contenido de una página y devuelve una lista de ellos.
    """
    internal_links = []
    # Aquí puedes implementar la lógica para encontrar los enlaces internos en el contenido HTML de la página
    # Por ahora, simularemos algunos enlaces internos
    internal_links.append({"url": "https://example.com/page1", "anchor_text": "Page 1"})
    internal_links.append({"url": "https://example.com/page2", "anchor_text": "Page 2"})
    internal_links.append({"url": "https://example.com/page3", "anchor_text": "Page 3"})
    return internal_links

def calculate_seo_recommendations(internal_links):
    """
    Calcula recomendaciones de SEO para los enlaces internos dados.
    """
    seo_recommendations = []
    for link in internal_links:
        anchor_text = link['anchor_text']
        url = link['url']
        keyword_density = calculate_keyword_density(anchor_text, ['keyword'])  # Reemplaza 'keyword' por tu palabra clave principal
        semantic_similarity = calculate_semantic_similarity(anchor_text, url)  # Calcula la similitud semántica entre el texto ancla y la URL
        recommendation = {"url": url, "anchor_text": anchor_text, "keyword_density": keyword_density, "semantic_similarity": semantic_similarity}
        seo_recommendations.append(recommendation)
    return seo_recommendations

def calculate_semantic_similarity(anchor_text, url):
    """
    Calcula la similitud semántica entre el texto ancla y la URL.
    Puedes implementar una lógica más avanzada aquí basada en el análisis de contenido.
    """
    # Por ahora, simplemente devolvemos una puntuación aleatoria entre 0 y 1
    return random.random()

async def get_processed_data_from_storage(domain: str, redis_client: Redis):
    # Aquí iría la lógica real para obtener los datos procesados del almacenamiento
    # Por ahora, simplemente devolvemos una lista vacía como placeholder
    return []


####################


#################### CONTENIDO DUPLICADO

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calcula la similitud entre dos textos utilizando el coeficiente de similitud de secuencia.
    """
    return SequenceMatcher(None, text1, text2).ratio()

async def find_duplicate_content(urls: List[str], client: httpx.AsyncClient, redis_client: Redis) -> Dict[str, List[str]]:
    """
    Encuentra contenido duplicado entre un conjunto de URLs.
    """
    duplicate_content = {}
    for i, url1 in enumerate(urls):
        for j, url2 in enumerate(urls):
            if i != j and (url1, url2) not in duplicate_content:
                # Verifica si ya hemos calculado la similitud entre estos dos URLs
                cached_similarity = await redis_client.get(f"duplicate_content:{url1}_{url2}")
                if cached_similarity:
                    similarity = float(cached_similarity)
                else:
                    # Si no está en el caché, obtén el contenido de ambas URL y calcula la similitud
                    content1, content2 = await fetch_page_content(client, url1), await fetch_page_content(client, url2)
                    similarity = calculate_similarity(content1, content2)
                    # Almacena la similitud en el caché para futuras referencias
                    await redis_client.set(f"duplicate_content:{url1}_{url2}", str(similarity), ex=86400)
                
                # Si la similitud supera un umbral, consideramos que el contenido es duplicado
                if similarity > 0.9:  # Umbral arbitrario
                    if url1 not in duplicate_content:
                        duplicate_content[url1] = [url2]
                    else:
                        duplicate_content[url1].append(url2)
    return duplicate_content

async def fetch_page_content(client: httpx.AsyncClient, url: str) -> str:
    """
    Obtiene el contenido de una URL.
    """
    try:
        response = await client.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch content for URL {url}, status code: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Error fetching content for URL {url}: {e}")
        return ""

@app.post("/analyze_duplicate_content")
async def analyze_duplicate_content(request: BatchRequest):
    try:
        urls = await fetch_sitemap_urls(request.domain)
        if not urls:
            raise HTTPException(status_code=404, detail="No URLs found in the sitemap.")
        
        duplicate_content = await find_duplicate_content(urls, app.state.client, app.state.redis)
        return duplicate_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

####################


#################### THIN CONTENT

async def analyze_thin_content(url_data):
    thin_content_urls = []
    for url, content_data in url_data.items():
        # Verificar si cumple con los criterios para ser considerado thin content
        if not content_data.get('h2_count') or content_data['h2_count'] <= 1:
            thin_content_urls.append(url)
        elif len(content_data.get('title', '')) < 30:
            thin_content_urls.append(url)
        elif len(content_data.get('meta_description', '')) < 80:
            thin_content_urls.append(url)
        elif not contains_semantic_words(content_data.get('slug', '')):
            thin_content_urls.append(url)
        elif not content_data.get('relevant_content'):
            thin_content_urls.append(url)
    return thin_content_urls

def contains_semantic_words(slug):
    # Podrías agregar lógica más sofisticada aquí según tus necesidades
    # Por ahora, simplemente verificamos si el slug contiene alguna letra
    return any(char.isalpha() for char in slug)

@app.post("/analyze_thin_content")
async def analyze_thin_content_endpoint(request: BatchRequest):
    try:
        urls = await fetch_sitemap_urls(request.domain)
        if not urls:
            raise HTTPException(status_code=404, detail="No URLs found in the sitemap.")
        
        url_data = await analyze_urls_content(urls, app.state.client, app.state.redis)
        thin_content_urls = await analyze_thin_content(url_data)
        return thin_content_urls
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

####################


#################### 404

@app.post("/handle_404_urls")
async def handle_404_urls(request: Request, domain: str, batch_size: Optional[int] = 50):
    redis_client: Redis = request.state.redis
    try:
        print(f"Handling 404 URLs for domain: {domain}")

        # Obtener todas las URLs que devolvieron un código de estado 404
        urls_to_handle = await get_404_urls(domain, batch_size, redis_client)
        if not urls_to_handle:
            raise HTTPException(status_code=404, detail=f"No 404 URLs found for domain: {domain}")

        # Procesar las URLs 404 encontradas según sea necesario (por ejemplo, notificar al administrador del sitio, guardarlas para su posterior revisión, etc.)
        # Aquí se puede agregar la lógica de manejo de URLs 404

        return {"message": f"{len(urls_to_handle)} 404 URLs found and processed for domain: {domain}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_404_urls(domain: str, batch_size: int, redis_client: Redis):
    redis_key = f"404_urls:{domain}"
    cached_urls = await redis_client.get(redis_key)
    if cached_urls:
        print(f"Cache hit for 404 URLs of domain {domain}")
        cached_data = json.loads(cached_urls)
        if not cached_data:
            print("Cached 404 URLs data is empty, refetching...")
        else:
            return cached_data  # Utilizar los datos del caché si están presentes y son válidos

    # Aquí puedes realizar una búsqueda en el almacenamiento de los datos generados por process_urls_in_batches para obtener las URLs 404 registradas para el dominio dado
    # Por ahora, simplemente devolveremos una lista vacía como si no se encontraran URLs 404
    return []

####################


#################### 301

async def suggest_301_redirects(urls_with_duplicates: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Sugiere redirecciones 301 para URLs con contenido duplicado.
    """
    redirections = {}
    for canonical_url, duplicate_urls in urls_with_duplicates.items():
        # Elige una URL duplicada para redireccionar
        url_to_redirect = duplicate_urls[0]  # Puedes elegir cualquier lógica para esto

        # Genera la regla de redirección 301
        redirection_rule = f"Redirect 301 {url_to_redirect} {canonical_url}"
        redirections[canonical_url] = redirection_rule

    return redirections

@app.post("/recommended_301_redirects")
async def recommended_301_redirects(request: Dict[str, List[str]]):
    """
    Endpoint para obtener recomendaciones de redirecciones 301.
    """
    try:
        urls_with_duplicates = request['urls_with_duplicates']
        suggested_redirects = await suggest_301_redirects(urls_with_duplicates)
        return suggested_redirects
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid request format. 'urls_with_duplicates' key not found.")

####################


#################### WPO

def analyze_wpo(url):
    try:
        # Realizar la solicitud HTTP para obtener el contenido de la página
        response = requests.get(url)
        response.raise_for_status()  # Lanza una excepción si la solicitud falla

        # Analizar el HTML de la página usando BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Ejemplo de métricas que podrías recopilar
        title = soup.title.string  # Título de la página
        page_size = len(response.content) / 1024  # Tamaño de la página en kilobytes
        load_time = response.elapsed.total_seconds()  # Tiempo de carga de la página en segundos

        # Puedes agregar más métricas según tus necesidades

        # Retornar las métricas recopiladas como un diccionario
        return {
            'title': title,
            'page_size_kb': page_size,
            'load_time_seconds': load_time
            # Agrega más métricas aquí si lo deseas
        }
    except Exception as e:
        raise RuntimeError(f"No se pudo analizar el WPO de {url}: {e}")

####################


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)