import requests
from bs4 import BeautifulSoup

url = 'http://aulacm.com'  # Reemplaza esto con la URL real de tu aplicación

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extraer y limpiar URLs
links = set()  # Usamos un conjunto para evitar duplicados
for link in soup.find_all('a', href=True):
    href = link['href']
    # Filtrar enlaces no deseados
    if not (href.startswith('#') or href.startswith('javascript')):
        links.add(href.strip())  # .strip() elimina espacios en blanco alrededor

# Imprimir URLs limpias
for clean_link in links:
    print(clean_link)

# Extraer y limpiar texto
text = soup.get_text()
clean_text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())  # Elimina líneas vacías y espacios extra
print(clean_text)
