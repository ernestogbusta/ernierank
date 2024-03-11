import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import json
import sys

class EnhancedRobotsParser:
    def __init__(self, url):
        self.url = url
        self.ua = UserAgent()
        self.headers = {'User-Agent': str(self.ua.random)}

    def fetch_and_parse(self):
        try:
            response = requests.get(self.url, headers=self.headers)
            if response.status_code == 200:
                content = BeautifulSoup(response.text, 'html.parser')
                return self.extract_data(content)
        except requests.exceptions.RequestException as e:
            print(f"Error al obtener {self.url}: {e}")

        return None

    def extract_data(self, content):
        data = {
            'h1': [h.text.strip() for h in content.find_all('h1')],
            'h2': [h.text.strip() for h in content.find_all('h2')],
            'description': [meta['content'].strip() for meta in content.find_all('meta', {'name': 'description'})],
            'title': content.find('title').text.strip() if content.find('title') else '',
            'body_text': self.fix_text(content.find('body').text.strip()) if content.find('body') else ''
        }

        return data

    def fix_text(self, text):
        # Reemplazar el texto problemático
        corrected_text = text.replace('\\n', ' ').replace('\n', ' ')
        return corrected_text

    def save_data(self, data):
        if data:
            with open('data.json', 'w') as json_file:
                json.dump(data, json_file, indent=2, ensure_ascii=False)
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
