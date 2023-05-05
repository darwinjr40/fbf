from flask import Flask, Response, Blueprint, render_template, request, jsonify, url_for
import cv2
import numpy as np
from werkzeug.utils import redirect
from werkzeug.exceptions import abort
import base64
import random


# import face_recognition as fr   #pip install face_recognition
import face_recognition as fr
import os
from datetime import datetime

camera_bp = Blueprint('camera', __name__)


# Definir la vista "inicio"
@camera_bp.route('/camera')
def camera():
    # return render_template('inicio.html')
    return 'camer'

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
# @camera_bp.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')            

@camera_bp.route("/inicio")
def video_feed():
    return "¡Hola, mundo!123"
    
@camera_bp.route('/prueba')
def prueba():
    return render_template('prueba.html')   

@camera_bp.route("/mostrar", methods=['GET', 'POST'])
def mostrar_nombre():
    # return "¡Hola, mundo!12"
    # return camera_bp.root_path 
    # return render_template('/home/hp-user/Documentos/python/service-flask/templates/index.html')
    return render_template('error404.html')