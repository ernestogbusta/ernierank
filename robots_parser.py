import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from reppy.robots import Robots
from fake_useragent import UserAgent
import json
import sys  # Importar el módulo sys

# Verificar si se proporcionó un argumento de URL
if len(sys.argv) > 1:
    url = sys.argv[1]
else:
    print("Por favor, proporciona una URL como argumento.")
    sys.exit(1)

ua = UserAgent()

def get_bot_loc(url):
    domain_url = '{uri.scheme}://{uri.netloc}'.format(uri=urlparse(url))
    bot_loc = domain_url + '/robots.txt'
    return bot_loc

def robot_parser(url):
    bot_loc = get_bot_loc(url)
    parser = Robots.fetch(bot_loc)
    validation = parser.allowed(url, '*')
    return validation

def fetch(url):
    header = {'User-Agent': str(ua.random)}
    validation = robot_parser(url)
    if validation:
        try:
            response = requests.get(url, headers=header)
            content = BeautifulSoup(response.text, 'html.parser')
            return content, response
        except requests.exceptions.ConnectionError:
            print(f'Error: "{url}" no está disponible.')
            return None, None
    else:
        print(f'{url} está bloqueado por robots.txt')
        return None, None

def parse(url):
    content, response = fetch(url)
    if response and response.status_code == 200:
        # Inicializar listas para almacenar datos extraídos
        h1, h2, description, title, canonical, hreflang, robot_tag = ([] for i in range(7))

        # Extracción de datos
        h1.extend([h.text.strip() for h in content.find_all('h1')])
        h2.extend([h.text.strip() for h in content.find_all('h2')])
        description.extend([tag['content'].strip() for tag in content.find_all('meta', {'name': 'description'})])
        title.extend([tag.text.strip() for tag in content.find_all('title')])
        canonical.extend([tag['href'].strip() for tag in content.find_all('link', {'rel': 'canonical'})])
        hreflang.extend([tag['href'].strip() for tag in content.find_all('link', {'rel': 'alternate'})])
        robot_tag.extend([tag['content'].strip() for tag in content.find_all('meta', {'name': 'robots'})])
        
        # Almacenar datos en un diccionario
        data = {
            'url': url,
            'h1': h1, 
            'h2': h2,
            'description': description,
            'title': title,
            'canonical': canonical,
            'hreflang': hreflang,
            'robot_tag': robot_tag 
        }
        print(data)

        # Escribir datos en archivo JSON
        write_to_file(data)
    elif response:
        print(f'No se pudo obtener {url}, código de estado: {response.status_code}')
    else:
        print(f'No se pudo obtener {url}')

def write_to_file(data):
    with open('data.json', 'w') as json_file:
        json.dump(data, json_file, indent=2)

parse(url)  # Llamar a la función parse con la URL como argumento
