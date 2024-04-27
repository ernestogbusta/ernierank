# analyze_wpo.py

import httpx
from bs4 import BeautifulSoup
import time
from fastapi import HTTPException

async def fetch_resource_size(client, resource_url):
    try:
        response = await client.head(resource_url, timeout=10)
        response.raise_for_status()
        size = int(response.headers.get('Content-Length', 0))
        return size
    except Exception as e:
        print(f"Failed to fetch size for {resource_url}: {str(e)}")
        return 0  # Retornar 0 si hay un error

async def analyze_resources(client, soup, tag_name, attr_name, size_threshold, base_url, filter_func=None):
    resources = []
    for tag in soup.find_all(tag_name):
        if filter_func and not filter_func(tag):
            continue
        src = tag.get(attr_name)
        if src:
            full_url = src if src.startswith('http') else f'{base_url}/{src.lstrip("/")}'
            size = await fetch_resource_size(client, full_url)
            if size > size_threshold:
                resources.append({"resource": full_url, "size_kb": round(size / 1024, 2)})
    return resources

async def analyze_wpo(url):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.RequestError as exc:
            print(f"An error occurred while requesting {exc.request.url!r}.")
            return {"error": "Server not available. Try again later."}
        
        soup = BeautifulSoup(response.content, 'html.parser')
        load_time = response.elapsed.total_seconds()
        content_length = len(response.content)
        
        base_url = url.rstrip('/')
        heavy_images = await analyze_resources(client, soup, 'img', 'src', 100 * 1024, base_url)
        heavy_scripts = await analyze_resources(client, soup, 'script', 'src', 100 * 1024, base_url)
        heavy_css = await analyze_resources(client, soup, 'link', 'href', 50 * 1024, base_url, lambda tag: 'stylesheet' in tag.get('rel', []))

        num_images, total_image_size = len(heavy_images), sum(img['size_kb'] for img in heavy_images)
        num_scripts, total_script_size = len(heavy_scripts), sum(script['size_kb'] for script in heavy_scripts)
        num_css, total_css_size = len(heavy_css), sum(css['size_kb'] for css in heavy_css)

        evaluation = interpret_wpo_metrics(
            load_time, content_length / 1024, num_images, total_image_size,
            num_scripts, total_script_size, num_css, total_css_size,
            heavy_images, heavy_scripts, heavy_css
        )

        wpo_metrics = {
            "url": url,
            "status_code": response.status_code,
            "load_time_seconds": load_time,
            "content_length_kb": round(content_length / 1024, 2),
            "images_size_warning": heavy_images,
            "scripts_size_warning": heavy_scripts,
            "css_size_warning": heavy_css,
            "evaluation": evaluation
        }

        return wpo_metrics

def interpret_wpo_metrics(load_time, content_length, num_images, total_image_size, num_scripts, total_script_size, num_css, total_css_size, heavy_images, heavy_scripts, heavy_css):
    evaluation = []
    
    # Evaluaciones de rendimiento basadas en el tiempo de carga
    if load_time < 1:
        evaluation.append("La velocidad de carga es rápida (< 1 segundo).")
    elif load_time < 3:
        evaluation.append("La velocidad de carga es moderada (< 3 segundos).")
    else:
        evaluation.append("La velocidad de carga es lenta (> 3 segundos).")

    # Evaluación del tamaño del contenido
    if content_length < 500:
        evaluation.append("El contenido de la web es ligero (< 500 KB).")
    elif content_length < 2000:
        evaluation.append("El contenido de la web es moderado (< 2 MB).")
    else:
        evaluation.append("El contenido de la web es pesado (> 2 MB).")

    # Evaluaciones de recursos pesados
    evaluation.append(f"{len(heavy_images)} imágenes pesadas detectadas (umbral: 100 KB).")
    evaluation.append(f"{len(heavy_scripts)} scripts pesados detectados (umbral: 50 KB).")
    evaluation.append(f"{len(heavy_css)} archivos CSS pesados detectados (umbral: 50 KB).")

    if not heavy_images:
        evaluation.append("No hay imágenes pesadas.")
    if not heavy_scripts:
        evaluation.append("No hay scripts pesados.")
    if not heavy_css:
        evaluation.append("No hay archivos CSS pesados.")

    return evaluation
