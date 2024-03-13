import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import json
import sys
from urllib.parse import urlparse
import os

class EnhancedRobotsParser:
    def __init__(self, url):
        self.url = url
        self.ua = UserAgent()
        self.headers = {'User-Agent': str(self.ua.random)}

    def fetch_and_parse(self):
        try:
            response = requests.get(self.url, headers=self.headers)
            if response.status_code == 404:
                print(f"Error 404: La página no se encontró - {self.url}")
                return None
            response.raise_for_status()  # Esto lanzará una excepción si el código de estado es 4XX o 5XX
            content = BeautifulSoup(response.text, 'html.parser')
            return self.extract_data(content)
        except requests.exceptions.HTTPError as e:
            print(f"Error HTTP: {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"Error de conexión: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Error al obtener {self.url}: {e}")
        return None

    def extract_data(self, content):
        # Tu lógica existente de extracción de datos
        data = {
            'h1': [h.text.strip() for h in content.find_all('h1')],
            'h2': [h.text.strip() for h in content.find_all('h2')],
            'description': [meta['content'].strip() for meta in content.find_all('meta', {'name': 'description'})],
            'title': content.find('title').text.strip() if content.find('title') else '',
            'body_text': self.fix_text(content.find('body').text.strip()) if content.find('body') else ''
        }
        return data

    def fix_text(self, text):
        # Tu lógica existente para limpiar el texto
        corrected_text = text.replace('\\n', ' ').replace('\n', ' ')
        return corrected_text

    def save_data(self, data):
        if data:
            parsed_url = urlparse(self.url)
            safe_filename = f"{parsed_url.netloc.replace('.', '_')}.json"
            filepath = os.path.join('data', safe_filename)
            with open(filepath, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=2, ensure_ascii=False)
            print(f"Datos guardados en {filepath}")
        else:
            print(f"No se extrajeron datos para guardar de {self.url}.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
        parser = EnhancedRobotsParser(url)
        data = parser.fetch_and_parse()
        if data:
            parser.save_data(data)
    else:
        print("Por favor, proporciona una URL como argumento.")
