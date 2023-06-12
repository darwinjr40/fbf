import sys
import multiprocessing
import time
import os

from flask import Flask
from flask_cors import CORS
sys.path.append('controllers')
from controller_service import service_bp
from controller_camera import camera_bp
#from controller_socket import create_socketio_app
from controllers.controller_socket import  socketio
from flask_sslify import SSLify
from flask_apscheduler import APScheduler
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
# sslify = SSLify(app, permanent=True, subdomains=True)
app.config['SECRET_KEY'] = 'secret!'# Configura la aplicación Flask para utilizar Flask-SocketIO
socketio.init_app(app)

    
# socketio = create_socketio_app(app)
# Registra la vista "inicio" en la aplicación Flask
app.register_blueprint(service_bp)
app.register_blueprint(camera_bp)

if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0')
    app.run( host='0.0.0.0')
    
    

