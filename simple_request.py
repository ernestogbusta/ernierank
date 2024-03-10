#!/usr/bin/env python
'''
Simple HTTP Get Request with User Input URL
$ pip install requests
'''
import requests
import sys

if len(sys.argv) < 2:
    print("Usage: python simple_request.py <URL>")
    sys.exit(1)

url = sys.argv[1]

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
}

try:
    r = requests.get(url, headers=headers)
    print(r.text)
    print(r.url)
    print(r.status_code)
except requests.exceptions.RequestException as e: 
    raise SystemExit(e)
