from bs4 import BeautifulSoup
from typing import List, Dict
import re
from urllib.parse import urlparse
from collections import Counter
import httpx

async def analyze_url(url: str, client: httpx.AsyncClient) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    try:
        response = await client.get(url, headers=headers)
        print(f"Attempting to process URL: {url} with status: {response.status_code}")
        if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
            soup = BeautifulSoup(response.content, 'html.parser')
            slug = extract_slug(url)
            title = soup.title.text if soup.title else "No title"
            meta_description = find_meta_description(soup)
            h1s = [h1.text.strip() for h1 in soup.find_all('h1')]
            h2s = [h2.text.strip() for h2 in soup.find_all('h2', limit=3)]
            text_relevant = ' '.join([p.text.strip() for p in soup.find_all('p', limit=5)])

            main_keyword = slug
            secondary_keywords = find_keywords(title, h1s, h2s, text_relevant, exclude=[slug])
            semantic_search_intent = calculate_semantic_search_intent(main_keyword, secondary_keywords)

            return {
                "url": url,
                "title": title,
                "meta_description": meta_description,
                "main_keyword": main_keyword,
                "secondary_keywords": secondary_keywords,
                "semantic_search_intent": semantic_search_intent
            }
        else:
            return None
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None


def calculate_semantic_search_intent(main_keyword, secondary_keywords):
    # Combine main and secondary keywords with given weights
    combined_keywords = [main_keyword] + secondary_keywords
    combined_text = ' '.join(combined_keywords)
    words = combined_text.split()
    
    # Eliminate duplicate words while preserving order
    seen = set()
    unique_words = [x for x in words if not (x in seen or seen.add(x))]
    
    # Ensure the length does not exceed 7 words
    return ' '.join(unique_words[:7])


def extract_relevant_text(soup):
    # Encuentra todos los encabezados principales y extrae texto significativo
    texts = []
    for header in soup.find_all(['h1', 'h2']):
        next_node = header.find_next_sibling()
        while next_node and next_node.name not in ['h1', 'h2', 'h3']:
            if next_node.name == 'p':
                texts.append(next_node.text)
            next_node = next_node.find_next_sibling()
    return ' '.join(texts)

# Define tus stopwords y términos genéricos
stopwords = {"y", "de", "la", "el", "en", "un", "una", "que", "es", "por", "con", "los", "las", "del", "al", "como", "para", "más", "pero", "su", "le", "lo", "se", "a", "o", "e", "nos", "sin", "sobre", "entre", "si", "tu", "mi", "te", "se", "qué", "cómo", "cuándo", "dónde", "por qué", "cuál", "su"}
generic_keywords = {"marketing digital", "community manager", "aula cm", "qué es", "para qué", "para qué sirve", "cómo funciona"}
common_verbs = {"aprender", "crear", "hacer", "usar", "instalar", "optimizar", "mejorar", "configurar", "desarrollar", "construir"}
adjectives = {"mejor", "nuevo", "grande", "importante", "primero", "último"}
question_starts = {"cómo", "qué", "para qué", "cuál es", "dónde"}
verb_synonyms = {
    "aprender": ["estudiar", "entender", "comprender", "capacitarse"],
    "usar": ["utilizar", "emplear", "manejar", "aplicar"],
    "mejorar": ["optimizar", "perfeccionar", "avanzar", "incrementar"],
    "crear": ["desarrollar", "generar", "producir", "construir"],
    "hacer": ["realizar", "ejecutar", "llevar a cabo", "confeccionar"],
    "instalar": ["configurar", "establecer", "implementar"],
    "optimizar": ["mejorar", "maximizar", "refinar"],
    "configurar": ["ajustar", "establecer", "programar"],
    "desarrollar": ["elaborar", "expandir", "crecer"],
    "construir": ["edificar", "fabricar", "montar"],
    "analizar": ["examinar", "evaluar", "revisar"],
    "promover": ["impulsar", "fomentar", "propagar"],
    "integrar": ["unificar", "incorporar", "combinar"],
    "lanzar": ["iniciar", "presentar", "introducir"],
    "gestionar": ["administrar", "manejar", "dirigir"],
    "innovar": ["renovar", "inventar", "pionear"],
    "optimizar": ["enhance", "boost", "elevate"],
    "navegar": ["explorar", "buscar", "revisar"],
    "convertir": ["transformar", "cambiar", "modificar"],
    "monetizar": ["rentabilizar", "capitalizar", "ingresar"],
    "segmentar": ["dividir", "clasificar", "categorizar"],
    "targetear": ["enfocar", "dirigir", "apuntar"],
    "publicar": ["difundir", "emitir", "divulgar"],
    "analizar": ["investigar", "sondear", "escrutar"],
    "comunicar": ["informar", "notificar", "transmitir"],
    "diseñar": ["esbozar", "planificar", "trazar"],
    "innovar": ["actualizar", "modernizar", "revolucionar"],
    "negociar": ["acordar", "tratar", "concertar"],
    "organizar": ["ordenar", "estructurar", "planear"],
    "planificar": ["programar", "proyectar", "prever"],
    "producir": ["fabricar", "generar", "crear"],
    "programar": ["codificar", "desarrollar", "planificar"],
    "promocionar": ["publicitar", "anunciar", "divulgar"],
    "recomendar": ["aconsejar", "sugerir", "proponer"],
    "reducir": ["disminuir", "decrementar", "minimizar"],
    "reforzar": ["fortalecer", "intensificar", "consolidar"],
    "registrar": ["anotar", "inscribir", "documentar"],
    "relacionar": ["vincular", "asociar", "conectar"],
    "remodelar": ["renovar", "reformar", "modernizar"],
    "rentabilizar": ["lucrar", "beneficiar", "capitalizar"],
    "replicar": ["imitar", "reproducir", "copiar"],
    "resolver": ["solucionar", "arreglar", "aclarar"],
    "responder": ["contestar", "reaccionar", "replicar"],
    "restaurar": ["reparar", "recuperar", "renovar"],
    "resultar": ["concluir", "derivarse", "provenir"],
    "retener": ["conservar", "mantener", "preservar"],
    "revelar": ["descubrir", "mostrar", "desvelar"],
    "revisar": ["examinar", "inspeccionar", "corregir"],
    "satisfacer": ["complacer", "contentar", "cumplir"],
    "segmentar": ["dividir", "separar", "clasificar"],
    "seleccionar": ["elegir", "escoger", "preferir"],
    "simplificar": ["facilitar", "clarificar", "depurar"],
    "sintetizar": ["resumir", "condensar", "abreviar"],
    "sistematizar": ["organizar", "ordenar", "estructurar"],
    "solucionar": ["resolver", "remediar", "arreglar"],
    "subrayar": ["enfatizar", "destacar", "resaltar"],
    "sugerir": ["proponer", "indicar", "recomendar"],
    "supervisar": ["controlar", "inspeccionar", "vigilar"],
    "sustituir": ["reemplazar", "cambiar", "suplantar"],
    "transformar": ["cambiar", "modificar", "alterar"],
    "transmitir": ["comunicar", "difundir", "emitir"],
    "valorar": ["evaluar", "apreciar", "estimar"],
    "variar": ["modificar", "alterar", "cambiar"],
    "vender": ["comercializar", "negociar", "ofertar"],
    "verificar": ["comprobar", "confirmar", "validar"],
    "viajar": ["trasladarse", "desplazarse", "moverse"],
    "vincular": ["enlazar", "asociar", "conectar"],
    "visitar": ["recorrer", "frecuentar", "acudir"],
    "visualizar": ["imaginar", "ver", "prever"]
}

def refine_keywords(text: str) -> List[str]:
    words = re.findall(r'\w+', text.lower())
    filtered_words = [word for word in words if word not in stopwords and word not in generic_keywords]
    keywords = []
    for word in filtered_words:
        found = False
        for verb, syn_list in verb_synonyms.items():
            if word in syn_list:
                keywords.append(verb)
                found = True
                break
        if not found:
            keywords.append(word)
    return list(set(keywords))

def calculate_keyword_density(text: str, keywords: List[str]) -> float:
    if not text:
        return 0.0
    words = re.findall(r'\w+', text.lower())
    word_count = Counter(words)
    total_words = sum(word_count.values())
    keyword_hits = sum(word_count[word] for word in keywords if word in word_count)
    if total_words == 0:
        return 0.0
    return (keyword_hits / total_words) * 100

def extract_slug(url: str) -> str:
    parsed_url = urlparse(url)
    path_segments = parsed_url.path.strip("/").split('/')
    return ' '.join(path_segments[-1].replace('-', ' ').split())

def find_keywords(title: str, h1s: List[str], h2s: List[str], text_relevant: str, exclude: List[str]) -> List[str]:
    """
    Identifies and returns the top secondary keywords from the provided content,
    excluding any specified words and prioritizing by relevance and length,
    ensuring they do not contain stop words or digits and are similar to the main keyword.
    """
    content = ' '.join([title] + h1s + h2s + [text_relevant])
    all_keywords = extract_keywords(content)
    # Remove any excluded keywords (typically the main keyword)
    filtered_keywords = [k for k in all_keywords if k not in exclude and not any(word in stopwords or word.isdigit() for word in k.split())]
    # Ensure keywords are similar to the main keyword or repeat the main if no good matches
    final_keywords = [k for k in filtered_keywords if is_similar(k, exclude[0])]
    if not final_keywords:
        final_keywords = [exclude[0]] * 2  # Use main keyword if no suitable secondary keywords
    return final_keywords[:2]  # Return top 2 secondary keywords based on the logic

def is_similar(keyword: str, main_keyword: str) -> bool:
    """
    Check if the keyword is similar to the main keyword. Here, you might implement
    more complex similarity checks based on your specific needs.
    """
    main_parts = set(main_keyword.split())
    keyword_parts = set(keyword.split())
    return len(main_parts & keyword_parts) > 0

def extract_keywords(text: str) -> List[str]:
    """
    Extracts keywords from the given text using n-gram analysis, considering only 2-6 word combinations.
    Filters out any combinations containing stop words or numbers.
    """
    words = re.findall(r'\w+', text.lower())
    n_grams = [' '.join(words[i:i+n]) for n in range(2, 7) for i in range(len(words)-n+1)]
    n_grams = [n_gram for n_gram in n_grams if not any(word in stopwords or word.isdigit() for word in n_gram.split())]
    return n_grams

def filter_and_weight_keywords(keywords: List[str]) -> List[str]:
    """
    Filters keywords by their length and applies weighting logic to prefer keywords
    of 2-3 words over 4-6 words.
    """
    weighted_keywords = {}
    for keyword in keywords:
        words = keyword.split()
        length = len(words)
        if length in (2, 3):
            weight = 5
        elif length == 4:
            weight = 4
        elif length in (5, 6):
            weight = 3
        else:
            continue  # Skip any keywords outside the 2-6 word range

        weighted_keywords[keyword] = weight

    # Sort keywords by weight and then alphabetically within the same weight
    sorted_keywords = sorted(weighted_keywords.items(), key=lambda x: (-x[1], x[0]))
    return [kw for kw, weight in sorted_keywords]

def find_meta_description(soup: BeautifulSoup) -> str:
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        return meta_tag["content"]
    return "No description available"

def get_processed_data():
    # Esta función debe devolver los datos ya procesados que incluyen la 'Semantic Search Intent'
    return [
        {
            "url": "https://aulacm.com/",
            "status": 200,
            "title": "Aula CM - Escuela de Marketing Digital",
            "meta_description": "Especialistas en WordPress, Blogs y Redes Sociales.",
            "main_keyword": "marketing digital",
            "secondary_keywords": ["cursos marketing digital", "agencia marketing digital", "escuela marketing"],
            "semantic_search_intent": "Explora cursos de marketing digital en Aula CM"
        }
        # Añade más diccionarios por cada URL procesada
    ]

def fetch_data_from_api():
    url = "https://ernierank-vd20.onrender.com/process_urls"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch data from API")

def format_data_for_presentation(data):
    formatted_output = []
    for item in data['processed_urls']:  # Asumiendo que esto coincide con tu estructura de respuesta JSON
        if item['status'] == 200:  # Asegurando que solo procesas URLs con status 200
            formatted_output.append(
                f"URL: {item['url']}\n"
                f"Title: {item['title']}\n"
                f"Meta Description: {item['meta_description']}\n"
                f"Main Keyword: {item['main_keyword']}\n"
                f"Secondary Keywords: {', '.join(item['secondary_keywords'])}\n"
                f"Semantic Search Intent: {item['semantic_search_intent']}\n"
            )
    return "\n".join(formatted_output)

def test_api():
    try:
        api_response = fetch_data_from_api()
        presentation_output = format_data_for_presentation(api_response)
        print(presentation_output)
    except Exception as e:
        print(str(e))

