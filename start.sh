#!/bin/bash
python -m nltk.downloader punkt
python -m nltk.downloader averaged_perceptron_tagger
python -m nltk.downloader brown
gunicorn main:app -b :10000 --timeout 300