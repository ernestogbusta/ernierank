from locust import HttpUser, between, task

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def load_main_page(self):
        self.client.get("/")

    @task(3)
    def scrape_site(self):
        # Asegúrate de reemplazar "https://www.example.com" con la URL que deseas rastrear y analizar
        self.client.post("/scrape", json={"url": "https://www.aulacm.com"})
