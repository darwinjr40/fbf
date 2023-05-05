from flask import Flask, Response, Blueprint, render_template, request, jsonify, url_for
import cv2
import numpy as np
from werkzeug.utils import redirect
from werkzeug.exceptions import abort
from camera import VideoCamera, Singleton
import sys
sys.path.append('controllers')


# import face_recognition as fr   #pip install face_recognition
import face_recognition as fr
from datetime import datetime

camera_bp = Blueprint('camera', __name__)

# Definir la vista "inicio"
@camera_bp.route('/camera')
def camera():
    # return render_template('inicio.html')
    return 'camer'

# def gen_frames():
def gen_frames(cam):
    try:        
        # camera = cv2.VideoCapture(0)
        while True:
                # success, frame = camera.read()
            # if not success:
            #     break
            # else:
                # ret, buffer = cv2.imencode('.jpg', frame)
                # frame = buffer.tobytes()
                # frame = cam.get_frame()
                # frame = camara.get_detect_faces()
                frame = camara.get_compare_faces()
                # Genera el stream de bytes del frame
                yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                print('dentro------------------------ -')
        print('se salio-------------------------')
        # camera.release()
    except Exception as e:    
        return jsonify({'result': 'errors', 'type': f"Tipo de excepción: {type(e)}", 'errors': f"Mensaje de error: {e}"})  
  

@camera_bp.route('/video_feed')
def video_feed():
    global camara
    camara = VideoCamera()
    # camara = Singleton.get_instance()
    # return Response(gen_frames(),                    
    return Response(gen_frames(camara),
                    mimetype='multipart/x-mixed-replace; boundary=frame')  

@camera_bp.route("/inicio")
def inicio():
    return "¡Hola, mundo!123"
    
@camera_bp.route('/prueba')
def prueba():
    return render_template('camera.html')   
