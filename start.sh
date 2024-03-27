#!/bin/bash
# Descarga los recursos de NLTK necesarios
python download_nltk_resources.py
# Inicia la aplicación Flask con Gunicorn
gunicorn main:app -b :10000
