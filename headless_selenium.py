#!/usr/bin/env python
import time
import sys
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

if len(sys.argv) > 1:
    url = sys.argv[1]
else:
    print("Por favor, proporciona una URL como argumento.")
    sys.exit(1)

options = Options()
options.headless = True

service = Service(executable_path='/usr/local/bin/geckodriver')
driver = webdriver.Firefox(service=service, options=options)

driver.get(url)
time.sleep(3)
t = driver.title
print(f'Title: {t}')

driver.quit()
