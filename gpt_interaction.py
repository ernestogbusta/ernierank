from api_client import hacer_solicitud_a_api

# Supongamos que esta función se llama cuando el usuario proporciona una URL
def manejar_interaccion_con_usuario(url_usuario):
    try:
        resultado = hacer_solicitud_a_api(url_usuario)
        # Procesa el resultado como necesites...
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        # Manejar el error adecuadamente
