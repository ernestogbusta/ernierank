from functools import wraps
from flask import Flask, request, jsonify, abort

API_KEY = "fba647b41ae2483bd9d4dc19bd90ab94"

app = Flask(__name__)

def require_apikey(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-Api-Key') and request.headers.get('X-Api-Key') == API_KEY:
            return view_function(*args, **kwargs)
        else:
            abort(401)
    return decorated_function

@app.route('/all')
@require_apikey
def tu_endpoint():
    return jsonify({"mensaje": "¡Has accedido con éxito usando una API Key!"})
