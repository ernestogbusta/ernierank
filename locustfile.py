from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def scrape_site(self):
        # Asegúrate de cambiar la URL de prueba por la que desees usar
        self.client.post("/scrape", json={"url": "https://www.example.com"})
