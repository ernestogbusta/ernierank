import requests
from bs4 import BeautifulSoup

class RSSReader:
    def __init__(self, rss_url):
        self.url = rss_url
        self.articles_dicts = []

    def fetch_rss(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        try:
            response = requests.get(self.url, headers=headers)
            if response.status_code == 200:
                self.parse_rss(response.text)
                return True
            else:
                print(f'Failed to fetch RSS feed: HTTP Status Code {response.status_code}')
                return False
        except Exception as e:
            print(f'Error fetching the URL: {self.url}\n{e}')
            return False

    def parse_rss(self, rss_content):
        soup = BeautifulSoup(rss_content, 'xml')  # Use 'xml' parser for better RSS handling
        articles = soup.findAll('item')
        self.articles_dicts = [
            {
                'title': article.find('title').text,
                'link': article.find('link').next_sibling.strip(),
                'description': article.find('description').text,
                'pubdate': article.find('pubDate').text
            } for article in articles
        ]

    def get_articles(self):
        return self.articles_dicts

if __name__ == '__main__':
    rss_url = 'https://www.jcchouinard.com/author/jean-christophe-chouinard/feed/'
    reader = RSSReader(rss_url)
    if reader.fetch_rss():
        articles = reader.get_articles()
        for article in articles:
            print(article['title'])
