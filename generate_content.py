# generate_content.py

from pydantic import BaseModel, HttpUrl
import httpx
import json
from aiocache import Cache
import os
import asyncio
import logging

# Configuración de la caché usando variables de entorno
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_URL = os.getenv('REDIS_URL', f'redis://{REDIS_HOST}:{REDIS_PORT}')
cache = Cache(Cache.REDIS, endpoint=REDIS_HOST, port=REDIS_PORT, namespace="main")

# Cargar la clave de la API de OpenAI desde las variables de entorno
api_key = os.getenv("OPENAI_API_KEY")
headers = {
    "Authorization": f"Bearer {api_key}"
}

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

async def set_cached_data(url, data, ttl=604800):
    if isinstance(data, dict):
        data = json.dumps(data)  # Convierte el diccionario a JSON string
    await cache.set(url, data, ttl)

async def get_cached_data(url):
    data = await cache.get(url)
    if data:
        try:
            return json.loads(data)  # Convierte JSON string a diccionario
        except json.JSONDecodeError:
            logging.error("Failed to decode cached JSON data.")
            return None
    return None

class ContentRequest(BaseModel):
    url: HttpUrl

async def call_openai_gpt4(prompt):
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.7
    }
    attempts = 0
    while attempts < 3:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            attempts += 1
            logging.warning(f"Timeout occurred: {str(e)} - Attempt {attempts}")
            await asyncio.sleep(2 ** attempts)  # Exponential backoff
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
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

async def generate_content_based_on_seo_data(url, client):
    data = await get_cached_data(url)
    if not data:
        data = await process_new_data(url, client)
    if not data:
        logging.error(f"No data available to generate content for {url}")
        return "No content available"
    # Generate content based on the processed data
    return f"Generated Content: {data['title']} - {data['meta_description']}"

    
    return generate_seo_content(processed_data)

def generate_seo_content(processed_data):
    if not isinstance(processed_data, dict):
        logging.error("Invalid data type for processed_data. Expected a dictionary.")
        raise TypeError("Processed data must be a dictionary")
    
    title = processed_data.get('title', 'Título no proporcionado')
    meta_description = processed_data.get('meta_description', 'Descripción no proporcionada')
    main_keyword = processed_data.get('main_keyword', 'Palabra clave principal no especificada')
    secondary_keywords = processed_data.get('secondary_keywords', [])
    semantic_search_intent = processed_data.get('semantic_search_intent', 'Intención de búsqueda semántica no especificada')

    # Construcción del contenido HTML o de texto
    content = f"<h1>{title}</h1>\n"
    content += f"<p><strong>Descripción:</strong> {meta_description}</p>\n"
    content += f"<p><strong>Palabra clave principal:</strong> {main_keyword}</p>\n"
    content += f"<p><strong>Intención de búsqueda semántica:</strong> {semantic_search_intent}</p>\n"

    if secondary_keywords:
        content += "<h2>Palabras clave secundarias</h2>\n<ul>\n"
        for keyword in secondary_keywords:
            content += f"<li>{keyword}</li>\n"
        content += "</ul>\n"

    content += "<h2>Análisis Detallado</h2>\n"
    content += "<p>Este análisis ayuda a comprender cómo se puede optimizar el contenido para satisfacer las necesidades de búsqueda de los usuarios.</p>\n"
    
    logging.debug(f"Generated content for URL with processed data: {processed_data}")
    return content


MIN_WORDS = 1000  # Establece tu mínimo deseado

def ensure_minimum_content(data):
    word_count = len(data.split())
    while word_count < MIN_WORDS:
        data += " " + generate_additional_content()  # Asumiendo que tienes una función que genera más contenido
        word_count = len(data.split())
    return data

def save_processed_data_to_file(data, file_path="progress.json"):
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info("Data saved to file successfully.")
    except Exception as e:
        logging.error(f"Failed to save data to file: {str(e)}")

def load_processed_data_from_file(file_path="progress.json"):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("No previous data found. Starting fresh.")
        return {}
    except json.JSONDecodeError:
        print("Failed to decode JSON. File might be corrupted.")
        return {}

async def process_new_data(url, client):
    # Simulación de la obtención y procesamiento de datos
    # Deberías implementar la lógica de extracción y procesamiento aquí
    processed_data = {
        "title": "Example Title",
        "meta_description": "Example Meta Description",
        "main_keyword": "Example Main Keyword",
        "secondary_keywords": ["Keyword 1", "Keyword 2"],
        "semantic_search_intent": "Buy"
    }
    await set_cached_data(url, processed_data)
    return processed_data

async def fetch_url_data(url, client, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            response = await client.get(url, timeout=10.0)
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
