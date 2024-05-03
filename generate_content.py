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

async def process_new_data(url, client):
    retries = 3
    attempt = 0
    while attempt < retries:
        try:
            response = await client.get(url, headers=headers, timeout=120.0)
            response.raise_for_status()
            logging.debug(f"HTTP Response Status: {response.status_code}, URL: {url}")

            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('title').text.strip() if soup.find('title') else 'Título no encontrado'
            meta_description = soup.find('meta', attrs={'name': 'description'})
            meta_description = meta_description['content'].strip() if meta_description else 'Descripción no proporcionada'
            h1 = soup.find('h1').text.strip() if soup.find('h1') else 'H1 no encontrado'

            keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
            if keywords_meta and keywords_meta['content']:
                keywords = [kw.strip() for kw in keywords_meta['content'].split(',')]
            else:
                keywords = [title.split()[0]] if title != 'Título no encontrado' else ['Palabra clave relevante']

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
            attempt += 1
            logging.error(f"Attempt {attempt}: HTTP error occurred while fetching data from {url}: {str(e)}")
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logging.error(f"Unexpected error occurred while processing data for {url}: {str(e)}")
            return None
    return None

async def generate_seo_content(processed_data, client):
    title = processed_data.get('title', 'Título no encontrado')
    semantic_intent = processed_data.get('semantic_search_intent', 'Intención no especificada')
    main_keyword = processed_data.get('main_keyword', 'Palabra clave principal no especificada')
    secondary_keywords = processed_data.get('secondary_keywords', [])

    # Configuración para generar descripciones más detalladas
    max_tokens_title = 100
    max_tokens_meta = 200
    max_tokens_h1 = 150
    max_tokens_h2 = 800

    # Generar título SEO
    title_prompt = (
        f"Genera un título SEO de hasta 100 caracteres centrado en {title if title != 'Título no encontrado' else main_keyword}, "
        f"que sea atractivo pero no sensacionalista ni use expresiones exageradas como vender más y cosas así ni tampoco verbos en imperativo de estilo publicitario y que sea pertinente para la intención de búsqueda '{semantic_intent}'."
    )
    seo_title = f"<title>{await call_openai_gpt4(title_prompt, client, max_tokens_title)}</title>"

    # Generar meta descripción
    meta_description_prompt = (
        f"Genera una meta descripción de hasta 200 caracteres que resuma el contenido del sitio usando '{main_keyword}' y "
        f"refleje la intención '{semantic_intent}' basada en '{title}'."
    )
    seo_meta_description = f"<meta name='description' content='{await call_openai_gpt4(meta_description_prompt, client, max_tokens_meta)}'>"

    # Generar H1
    h1_prompt = (
        f"Genera un H1 que sea directamente relevante para el contenido y que utilice '{main_keyword}' como enfoque principal."
    )
    seo_h1 = f"<h1>{await call_openai_gpt4(h1_prompt, client, max_tokens_h1)}</h1>"

    full_content = f"{seo_title}\n{seo_meta_description}\n{seo_h1}\n"

    # Generar contenido para al menos cinco H2 con verificación de longitud
    required_paragraphs = 3
    min_words_per_paragraph = 200
    for i, keyword in enumerate([main_keyword] + secondary_keywords[:4]):  # Asegura al menos cinco H2 si es posible
        for _ in range(required_paragraphs):
            h2_prompt = (
                f"Genera un párrafo detallado sobre '{keyword}', asegurando un mínimo de {min_words_per_paragraph} palabras."
            )
            paragraph_content = await call_openai_gpt4(h2_prompt, client, max_tokens_h2)
            # Verificar si el párrafo cumple con el mínimo de palabras
            while len(paragraph_content.split()) < min_words_per_paragraph:
                additional_content = await call_openai_gpt4(h2_prompt, client, max_tokens_h2)
                paragraph_content += " " + additional_content
            full_content += f"<h2>{keyword}</h2>\n<p>{paragraph_content}</p>\n"

    return full_content

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
        "temperature": 0.7
    }
    try:
        response = await client.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()  # Raises an httpx.HTTPStatusError if the response has an HTTP error status.
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as exc:
        logging.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
        raise
    except httpx.RequestError as exc:
        logging.error(f"Request error occurred: {exc}")
        raise
    except KeyError as exc:
        logging.error(f"Unexpected response structure: {exc} - {response.text}")
        raise

async def fetch_processed_data(url: str, client: httpx.AsyncClient, progress_file: str):
    logging.debug(f"Fetching processed data for URL: {url} from {progress_file}")
    try:
        with open(progress_file, "r") as file:
            progress_data = json.load(file)
        for item in progress_data.get("urls", []):
            if item["url"] == url:
                logging.debug(f"Processed data found for URL: {url}")
                return item
        logging.warning(f"No processed data found for URL: {url}")
    except FileNotFoundError:
        logging.error(f"Progress file {progress_file} not found.")
        return {}
    except json.JSONDecodeError:
        logging.error("JSON decoding failed.")
        return {}
    return {}

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