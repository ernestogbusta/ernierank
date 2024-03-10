#!/usr/bin/env python
'''
Opens Firefox
Visit URL provided by the user
Print the HTML of the page
Close browser.
'''
import time
import sys
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

# Check if a URL has been provided as an argument
if len(sys.argv) > 1:
    url = sys.argv[1]
else:
    print("Please provide a URL as an argument.")
    sys.exit(1)

options = Options()
options.headless = True  # Uncomment if you want Firefox to run headlessly

service = Service(executable_path='/usr/local/bin/geckodriver')
driver = webdriver.Firefox(service=service, options=options)

driver.get(url)             # Visit URL provided by the user

print(driver.page_source)   # Print HTML of the page
time.sleep(3)               # Wait 3 Seconds
driver.quit()               # Close the browser
