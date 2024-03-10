import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from reppy.robots import Robots
from fake_useragent import UserAgent
import json
import sys

class EnhancedRobotsParser:
    def __init__(self, url):
        self.url = url
        self.ua = UserAgent()
        self.headers = {'User-Agent': str(self.ua.random)}

    def get_bot_loc(self):
        parsed_url = urlparse(self.url)
        domain_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        return f"{domain_url}/robots.txt"

    def is_allowed(self):
        bot_loc = self.get_bot_loc()
        try:
            parser = Robots.fetch(bot_loc)
            return parser.allowed(self.url, self.ua.random)
        except Exception as e:
            print(f"Error al verificar robots.txt: {e}")
            # Si hay un error al obtener/analizar robots.txt, procedemos con precaución
            return False

    def fetch_and_parse(self):
        if not self.is_allowed():
            print(f'{self.url} está bloqueado por robots.txt')
            return None

        try:
            response = requests.get(self.url, headers=self.headers)
            if response.status_code == 200:
                content = BeautifulSoup(response.text, 'html.parser')
                return self.extract_data(content)
        except requests.exceptions.RequestException as e:
            print(f"Error al obtener {self.url}: {e}")

        return None

    def extract_data(self, content):
        # Realiza la extracción de datos como en el código original
        data = {
            'h1': [h.text.strip() for h in content.find_all('h1')],
            'h2': [h.text.strip() for h in content.find_all('h2')],
            'description': [meta['content'].strip() for meta in content.find_all('meta', {'name': 'description'})],
            'title': content.find('title').text.strip() if content.find('title') else '',
            # Continúa extrayendo otros datos necesarios
        }
        return data

    def save_data(self, data):
        if data:
            with open('data.json', 'w') as json_file:
                json.dump(data, json_file, indent=2)
            print("Datos guardados en data.json")
        else:
            print("No se extrajeron datos para guardar.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
        parser = EnhancedRobotsParser(url)
        data = parser.fetch_and_parse()
        parser.save_data(data)
    else:
        print("Por favor, proporciona una URL como argumento.")
