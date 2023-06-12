import os
# from app import ENV
from dotenv import load_dotenv

load_dotenv() # Cargar variables de entorno desde el archivo .env

class ENV:
    DIR_FACES = os.getenv('DIR_FACES', 'personal' )
    