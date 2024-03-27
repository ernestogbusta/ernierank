import nltk

def download_nltk_resources():
    # Descarga solo los recursos específicos que necesites
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')  # Agrega esto si lo necesitas

if __name__ == "__main__":
    download_nltk_resources()
