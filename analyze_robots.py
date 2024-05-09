from fastapi import HTTPException
import httpx
from pydantic import BaseModel, HttpUrl

class RobotsTxtRequest(BaseModel):
    url: HttpUrl  # Usa HttpUrl para asegurar que se proporcione una URL válida

async def fetch_robots_txt(url: str):
    """Esta función asincrónica recupera el archivo robots.txt de un dominio dado."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{url.rstrip('/')}/robots.txt")
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail="Failed to fetch robots.txt")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching robots.txt: {str(e)}")

def analyze_robots_txt(content: str):
    """Analiza el contenido de robots.txt y extrae las reglas específicas que afectan al rastreo."""
    rules = {}
    user_agent = None
    for line in content.splitlines():
        if line.startswith('User-agent:'):
            user_agent = line.split(":")[1].strip()
            rules[user_agent] = []
        elif line.startswith(('Allow:', 'Disallow:')) and user_agent:
            rule = line.split(":", 1)
            rules[user_agent].append({rule[0].strip(): rule[1].strip()})
    return rules