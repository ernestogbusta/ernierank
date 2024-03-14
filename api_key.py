from functools import wraps
from flask import Flask, request, jsonify, abort

# Tu API key secreta
API_KEY = "fba647b41ae2483bd9d4dc19bd90ab94"

app = Flask(__name__)

def require_apikey(view_function):
    """Decorador que requiere que las solicitudes incluyan una API key válida."""
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        # Verificar si la API key proporcionada en los headers coincide con la API key esperada
        if request.headers.get('X-Api-Key') and request.headers.get('X-Api-Key') == API_KEY:
            return view_function(*args, **kwargs)
        else:
            # Abortar la solicitud con un error 401 si la API key no es válida o no se proporciona
            abort(401, description="API Key no válida o no proporcionada.")
    return decorated_function

@app.route('/all', methods=['GET'])
@require_apikey
def tu_endpoint():
    """Un ejemplo de endpoint protegido por API key."""
    # Retornar un mensaje JSON exitoso si la API key es válida
    return jsonify({"mensaje": "¡Has accedido con éxito usando una API Key!"})

if __name__ == "__main__":
    app.run(debug=True)
