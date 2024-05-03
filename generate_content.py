# generate_content.py

from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl
import httpx
from httpx import HTTPStatusError, RequestError
import json
import os
import asyncio
import logging
from urllib.parse import urlparse
import functools
import time

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Authorization": f"Bearer {api_key}"
}

class ContentRequest(BaseModel):
    url: HttpUrl

def timing_decorator(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Tiempo de ejecución de {func.__name__}: {end_time - start_time:.4f} segundos")
        return result
    return wrapper

@timing_decorator
async def process_new_data(url, client):
    retries = 3
    attempt = 0
    while attempt < retries:
        try:
            # Se utiliza el método get con reutilización de la conexión cliente y manejo eficiente del timeout
            response = await client.get(url, headers=headers, timeout=httpx.Timeout(10, connect=5))
            response.raise_for_status()
            logging.debug(f"HTTP Response Status: {response.status_code}, URL: {url}")

            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('title').text.strip() if soup.find('title') else 'Título no encontrado'
            meta_description = soup.find('meta', attrs={'name': 'description'})
            meta_description = meta_description['content'].strip() if meta_description else 'Descripción no proporcionada'
            h1 = soup.find('h1').text.strip() if soup.find('h1') else 'H1 no encontrado'

            keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
            keywords = [kw.strip() for kw in keywords_meta['content'].split(',')] if keywords_meta and keywords_meta['content'] else [title.split()[0]] if title != 'Título no encontrado' else ['Palabra clave relevante']

            main_keyword = keywords[0]
            secondary_keywords = keywords[1:]
            semantic_search_intent = 'Informar' if 'informar' in meta_description.lower() else 'Vender'

            processed_data = {
                "title": title,
                "meta_description": meta_description,
                "h1": h1,
                "main_keyword": main_keyword,
                "secondary_keywords": secondary_keywords,
                "semantic_search_intent": semantic_search_intent,
                "url": url
            }

            logging.info(f"Data processed for URL: {url}")
            return processed_data
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP Status Error on attempt {attempt+1} for URL {url}: {str(e)}")
            if e.response.status_code < 500:
                break  # No reintentar para errores de cliente 4XX
            attempt += 1
            await asyncio.sleep(min(60, 2 ** attempt))  # Exponential backoff con un máximo de 60 segundos
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logging.error(f"Network error on attempt {attempt+1} for URL {url}: {str(e)}")
            attempt += 1
            if attempt >= retries:
                logging.error("Max retry attempts reached, could not fetch data.")
                return None
            await asyncio.sleep(min(60, 2 ** attempt))  # Exponential backoff con un máximo de 60 segundos
        except Exception as e:
            logging.error(f"Unexpected error occurred while processing data for URL {url}: {str(e)}")
            return None

@timing_decorator
async def generate_seo_content(processed_data, client):
    title = processed_data.get('title', 'Título no encontrado')
    semantic_intent = processed_data.get('semantic_search_intent', 'Intención no especificada')
    main_keyword = processed_data.get('main_keyword', 'Palabra clave principal no especificada')
    secondary_keywords = processed_data.get('secondary_keywords', [])

    # Configuración para generar descripciones más detalladas
    max_tokens_title = 150
    max_tokens_meta = 200
    max_tokens_h1 = 150
    max_tokens_h2 = 900

    # Utilizar tareas asíncronas para realizar las llamadas API en paralelo cuando sea posible
    tasks = [
        call_openai_gpt4(
            f"Dame título SEO de 60 caracteres sobre {title if title != 'Título no encontrado' else main_keyword}, "
            f"atractivo pero que no sensacionalista ni verbos imperativos y sobre este tema '{semantic_intent}'.",
            client, max_tokens_title),
        call_openai_gpt4(
            f"Dame meta descripción de 155 caracteres sobre '{main_keyword}' y "
            f"y '{semantic_intent}' basada en '{title}'.",
            client, max_tokens_meta),
        call_openai_gpt4(
            f"Dame H1 relevante sobre '{main_keyword}'",
            client, max_tokens_h1)
    ]
    results = await asyncio.gather(*tasks)
    seo_title, seo_meta_description, seo_h1 = (f"<title>{results[0]}</title>",
                                               f"<meta name='description' content='{results[1]}'>",
                                               f"<h1>{results[2]}</h1>")

    full_content = f"{seo_title}\n{seo_meta_description}\n{seo_h1}\n"

    # Generar contenido para H2 y párrafos asociados
    for keyword in [main_keyword] + secondary_keywords[:4]:  # Limita a cinco H2
        h2_content = ""
        for _ in range(3):  # Tres párrafos por H2
            paragraph_content = await call_openai_gpt4(
                f"Dame un párrafo sobre '{keyword}', con un mínimo de {min_words_per_paragraph} palabras.",
                client, max_tokens_h2)
            h2_content += f"<p>{paragraph_content}</p>\n"
        full_content += f"<h2>{keyword}</h2>\n{h2_content}"

    return full_content

@timing_decorator
async def call_openai_gpt4(prompt, client, max_tokens):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    
    retries = 3
    backoff_factor = 2.0  # Factor de backoff exponencial
    for attempt in range(retries):
        try:
            # Ajusta los tiempos de conexión y lectura, aumentando en cada intento
            timeout_config = httpx.Timeout(10.0 + 10.0 * attempt, connect=5.0 + 5.0 * attempt)
            response = await client.post(url, json=payload, headers=headers, timeout=timeout_config)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.TimeoutException as e:
            logging.error(f"Timeout on attempt {attempt + 1}: {str(e)}")
            if attempt >= retries - 1:
                raise HTTPException(status_code=408, detail="Request to OpenAI timed out after multiple retries")
            await asyncio.sleep(backoff_factor ** attempt)  # Exponential backoff
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP status error on attempt {attempt + 1}: {str(e)}")
            raise HTTPException(status_code=e.response.status_code, detail=str(e.response.text))
        except httpx.RequestError as e:
            logging.error(f"Network request error on attempt {attempt + 1}: {str(e)}")
            if attempt >= retries - 1:
                raise HTTPException(status_code=500, detail="Network request failed at all retry attempts")
            await asyncio.sleep(backoff_factor ** attempt)  # Exponential backoff
        except Exception as e:
            logging.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt >= retries - 1:
                raise HTTPException(status_code=500, detail="An unexpected error occurred")
            await asyncio.sleep(backoff_factor ** attempt)  # Exponential backoff

async def fetch_url_data(url, client, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            response = await client.get(url, timeout=120.0)
            response.raise_for_status()
            return response.text
        except httpx.RequestError as e:
            logging.warning(f"Attempt {attempt+1}: Network error occurred while fetching data from {url}: {str(e)}")
            attempt += 1
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            break
    logging.error(f"Failed to fetch data from {url} after {retries} attempts.")
    return None

