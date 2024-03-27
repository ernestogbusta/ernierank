import nltk

def download_nltk_resources():
    # Descarga el tokenizador 'punkt'
    nltk.download('punkt')
    # Descarga el etiquetador 'averaged_perceptron_tagger'
    nltk.download('averaged_perceptron_tagger')

if __name__ == "__main__":
    download_nltk_resources()
