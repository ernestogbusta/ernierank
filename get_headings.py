import requests
from bs4 import BeautifulSoup
import sys

def fetch_headings(url):
    """
    Fetch and print headings (h1, h2, h3) from a given URL.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            headings = {'h1': [], 'h2': [], 'h3': []}
            for heading in soup.find_all(['h1', 'h2', 'h3']):
                headings[heading.name].append(heading.text.strip())
            return headings
        else:
            print(f'Failed to fetch the page: HTTP Status Code {response.status_code}')
    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')

def main(url):
    """
    Main function to handle command line arguments and call fetch_headings.
    """
    headings = fetch_headings(url)
    if headings:
        for tag, texts in headings.items():
            for text in texts:
                print(f'{tag}: {text}')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a URL as an argument.")
