import secrets

api_key = secrets.token_hex(16)  # Genera una cadena hexadecimal segura de 16 bytes
print(api_key)
