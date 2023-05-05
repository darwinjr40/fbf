from flask import Flask
from flask_cors import CORS
import sys
sys.path.append('controllers')

from controller_service import service_bp
from controller_camera import camera_bp
from controller_socket import create_socketio_app
# from flask_socketio import SocketIO, emit
from flask_sslify import SSLify



app = Flask(__name__)
sslify = SSLify(app, permanent=True, subdomains=True)

CORS(app)
app.config['SECRET_KEY'] = 'secret!'
socketio = create_socketio_app(app)

# Registra la vista "inicio" en la aplicación Flask
app.register_blueprint(service_bp)
app.register_blueprint(camera_bp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

