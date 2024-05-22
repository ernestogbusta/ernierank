# Este es el archivo analyze_404.py, dame código robusto y completo para solucionar el problema pero dame ÚNICAMENTE las funciones o endpoints que debo modificar o actualizar, EN NINGUN CASO me des funciones o endpoints que ya estén funcionando bien en mi código

from fastapi import HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
import httpx
from httpx import RemoteProtocolError
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import asyncio
import xmltodict
import requests
import httpx


async def fetch_urls(base_url, client):
    try:
        r = await client.get(base_url)
        r.raise_for_status()  # Esto asegura que solo procesamos respuestas exitosas
        soup = BeautifulSoup(r.content, 'html.parser')
        links = soup.find_all('a')
        urls = {urljoin(base_url, link.get('href')) for link in links if link.get('href')}
        return urls
    except httpx.RequestError as e:
        print(f"Request error: {e}")
        return set()

async def check_url(url, client):
    try:
        response = await client.get(url)
        return {'url': url, 'status': response.status_code}
    except httpx.RequestError as e:
        return {'url': url, 'status': 'error', 'error': str(e)}

async def crawl_site(start_url):
    client = httpx.AsyncClient()
    urls_to_check = await fetch_urls(start_url, client)
    results = await asyncio.gather(*(check_url(url, client) for url in urls_to_check))
    await client.aclose()
    return results

async def find_broken_links(domain: str) -> List[str]:
    broken_links = []
    try:
        # Configuración del cliente HTTP con timeout para evitar esperas prolongadas
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(domain)
            response.raise_for_status()  # Asegura que la respuesta es exitosa

            # Analiza el HTML para encontrar todos los enlaces
            soup = BeautifulSoup(response.text, 'html.parser')
            links = {urljoin(domain, link.get('href')) for link in soup.find_all('a') if link.get('href')}

            # Verifica cada enlace encontrado
            if links:
                check_responses = await asyncio.gather(*(client.get(link) for link in links))
                broken_links = [link for link, resp in zip(links, check_responses) if resp.status_code == 404]

            return broken_links

    except httpx.RequestError as e:
        print(f"Error en la solicitud HTTP: {str(e)}")
        raise HTTPException(status_code=500, detail="Error en la conexión con el servidor.")

    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail="Un error inesperado ocurrió.")

    return broken_links

