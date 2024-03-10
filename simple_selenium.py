import time
import sys
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

def open_firefox_and_print_html(url, headless=True):
    """
    Abre Firefox con la URL proporcionada, imprime el HTML de la página y cierra el navegador.
    
    Args:
    - url: URL de la página a visitar.
    - headless: Si True, ejecuta Firefox en modo headless (sin interfaz gráfica).
    """
    options = Options()
    if headless:
        options.add_argument("--headless")
    
    try:
        service = Service(executable_path='/usr/local/bin/geckodriver')
        driver = webdriver.Firefox(service=service, options=options)
        driver.get(url)
        
        # Esperar a que la página se cargue completamente. Ajustar según necesidad.
        time.sleep(3)
        
        print(driver.page_source)
    except Exception as e:
        print(f"Error al abrir la página: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        open_firefox_and_print_html(sys.argv[1])
    else:
        print("Por favor, proporciona una URL como argumento.")
