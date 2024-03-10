from lxml import html
import requests
import sys

if len(sys.argv) > 1:
    url = sys.argv[1]
else:
    print("Por favor, proporciona una URL como argumento.")
    sys.exit(1)

page = requests.get(url)
webpage = html.fromstring(page.content)

links = webpage.xpath('//a/@href')

print("Links obtenidos exitosamente:")
for link in links:
    print(link)
