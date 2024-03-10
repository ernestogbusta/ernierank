import sys
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

def get_page_title(url):
    """
    Opens a URL using Firefox in headless mode and returns the page title.
    """
    options = Options()
    options.headless = True
    service = Service(executable_path='/usr/local/bin/geckodriver')
    
    try:
        driver = webdriver.Firefox(service=service, options=options)
        driver.get(url)
        title = driver.title
        driver.quit()
        return title
    except Exception as e:
        print(f"Error opening page: {e}")
        return None

def main(url):
    """
    Main function to handle command line arguments and call get_page_title.
    """
    title = get_page_title(url)
    if title:
        print(f'Title: {title}')
    else:
        print("Failed to retrieve the page title.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a URL as an argument.")
