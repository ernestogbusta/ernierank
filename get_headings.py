#!/usr/bin/env python
'''
Print headings in a page
$ pip install requests
$ pip install bs4
'''
from bs4 import BeautifulSoup
import requests
import sys  # Importa sys para acceder a los argumentos de la línea de comandos

# Verifica que el usuario haya proporcionado una URL como argumento
if len(sys.argv) > 1:
    url = sys.argv[1]  # Toma la URL del primer argumento de la línea de comandos
else:
    print("Por favor, proporciona una URL como argumento.")
    sys.exit(1)  # Sale del script si no se proporciona ninguna URL

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
}

try:
    r = requests.get(url, headers=headers)
except requests.exceptions.RequestException as e: 
    raise SystemExit(e)

page = BeautifulSoup(r.text, "lxml")
for h in page.find_all(['h1','h2','h3']):
    print(f'{h.name}: {h.text.strip()}')
