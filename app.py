from flask import Flask
from flask_cors import CORS
import sys
sys.path.append('controllers')

from controller_service import service_bp
from controller_camera import camera_bp
# from controller_socket import create_socketio_app
from controllers.controller_socket import create_socketio_app

from flask_socketio import SocketIO, emit
from flask_sslify import SSLify
from flask_apscheduler import APScheduler
from apscheduler.schedulers.background import BackgroundScheduler



app = Flask(__name__)
# sslify = SSLify(app, permanent=True, subdomains=True)

# CORS(app)
app.config['SECRET_KEY'] = 'secret!'
socketio = create_socketio_app(app)
# socketio = SocketIO(app, cors_allowed_origins="*")

# Registra la vista "inicio" en la aplicación Flask
app.register_blueprint(service_bp)
app.register_blueprint(camera_bp)






# def my_job():
#    # Esta es la tarea que se ejecutará con el intervalo de tiempo específico
#    print("Hello, Flask APScheduler!")
#    start_repeating_task()
# scheduler = BackgroundScheduler()
# scheduler.add_job(id='job', func=my_job, trigger='date')
# # scheduler.add_job(id='job', func=my_job, trigger='date', run_date='2023-05-01 12:00:00')
# scheduler.start()
   
# scheduler = APScheduler()
# app.config['SCHEDULER_API_ENABLED'] = True
# scheduler.add_job(id = 'job', func = my_job, trigger = 'interval', seconds = 1)    
# scheduler.init_app(app)
# scheduler.start()
# Establecer la tarea con un intervalo de 5 segundos

if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0')
    app.run( host='0.0.0.0')
    

