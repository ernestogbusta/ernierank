#!/bin/bash
python download_nltk_resources.py
gunicorn main:app -b :10000
