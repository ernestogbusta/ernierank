from selenium import webdriver

# Esto asume que `geckodriver` ya está en tu PATH y accesible
driver = webdriver.Firefox()

# Abre una página web (por ejemplo, Google)
driver.get("http://www.google.com")

# Imprime el título de la página para verificar que se cargó correctamente
print(driver.title)

# Cierra el navegador
driver.quit()
