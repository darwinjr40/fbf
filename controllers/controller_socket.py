from collections import deque
import  os, cv2, base64, random, time
import multiprocessing
import asyncio
import math

import numpy as np
import face_recognition as fr
import cvlib as cv


from flask import Flask, Blueprint, render_template, request, jsonify, url_for
from werkzeug.utils import redirect
from werkzeug.exceptions import abort
from flask_socketio import SocketIO, emit
from io import BytesIO
from PIL import Image
from threading import Thread
from controllers.clases.hilo import  Hilo
from controllers.clases.colors import  Color
from controllers.clases.videos import  Video
from concurrent.futures import ThreadPoolExecutor
from config import ENV
from multiprocessing import Process, Queue
from keras.models import load_model
# from app import socketio
#-----------------------------------------------------
def saved_files(dir):
    global names_files_faces, images_face, names_files_complete_faces               
    lista = os.listdir(dir) # lista = os.listdir('.')    
    for data in lista:        #Leemos los rostros del DB
        imgdb = cv2.imread(f'{dir}/{data}') #Leemos Las imagenes de los rostros
        images_face.append(imgdb) # ALmacenamos imagen
        names_files_faces.append(os.path.splitext(data)[0]) #ALmacenamos nombre
    print(f"lista: {lista}")        
    print (f"names_files_faces: {names_files_faces}")
    


#Funcion para codificar los rostros
def codrostros(images):
    listacod = []
    for img in images: # Iterano0s        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Correccion de color        
        cod = fr.face_encodings(img)[0] #Codificamos la imagen        
        listacod.append(cod) #ALmacenamos
    return listacod


def saved_files1(dir): # Obtener la lista de archivos previamente procesados (si existe)
    global names_files_faces, coded_faces, names_files_complete_faces               
    previous_files = set(names_files_complete_faces) #names_files_complete_faces    
    current_files = set(os.listdir(dir)) # Obtener la lista actual de archivos en el directorio     
    new_files = current_files - previous_files # Identificar los nuevos archivos (diferencia entre las listas)
    for data in new_files:  
        imgdb = cv2.imread(f'{dir}/{data}')
        img = cv2.cvtColor(imgdb, cv2.COLOR_BGR2RGB) # Correccion de color        
        cod = fr.face_encodings(img)[0] #Codificamos la imagen        
        coded_faces.append(cod) #almacenamos
        names_files_faces.append(os.path.splitext(data)[0]) #ALmacenamos nombre          
    names_files_complete_faces = list(current_files) # Actuali zar la lista de archivos previamente procesados
    print (f"names_files_complete_faces: {names_files_complete_faces}")
    print (f"names_files_faces: {names_files_faces}")
    print (f"new_files: {new_files}")
    print (f"coded_faces: {coded_faces}")


#Main -----------------------------------------------------
#socketio = SocketIO(None, cors_allowed_origins="*", logger=True, engineio_logger=True, ping_timeout=300)
socketio = SocketIO(None, cors_allowed_origins="*")

names_files_complete_faces = []
names_files_faces = []
coded_faces = []
saved_files1(dir=ENV.DIR_FACES)
print(ENV.DIR_FACES)
sw_hilos = False
clients = 0
thread0 = None
thread1 = None
thread2 = None
thread3 = None
# hilo: Hilo = None
hilos : Hilo = [None , None]
dim_hilos = len(hilos)
images_face = []
queue = Queue(maxsize=256)  
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# saved_files(dir='personal')
# coded_faces = codrostros(images_face)




#get instancia------------------------------------------------------
def create_socketio_app(app):
    socketio.init_app(app)
    return socketio

#socket default------------------------------------------------------
@socketio.on('connect')
def handle_connect():
    global clients
    global queue
    clients = len(socketio.server.manager.rooms['/'].keys()) - 1  # Restar 1 para excluir al propio cliente
    print(f"se conectaron, Número de clientes conectados: {clients}")
    print(queue.qsize())
    # start_video_thread()
    # emit('processed_webrtc', 'asdasdsd')
    
@socketio.on('disconnect')
def handle_connect():
    global clients
    clients -=1
    print(f"se desconectaron, Número de clientes conectados: {clients}")
    # emit('processed_webrtc', 'asdasdsd')
    # start_repeating_task()
    

#socket eventos---------------------------------------------------------

@socketio.on('start')
def event(json):   
    global sw_hilos 
    saved_files1(dir=ENV.DIR_FACES)                
    sw_hilos = True    
    start_video_thread()


@socketio.on('restart')
def event(json):
    global sw_hilos
    sw_hilos = False
    time.sleep(2)
    saved_files1(dir=ENV.DIR_FACES)            
    sw_hilos = True        
    start_video_thread()

@socketio.on('stop')
def event(json):    
    global sw_hilos
    sw_hilos = False
    
@socketio.on('event')
def event(json):
    start_video_thread()
    # print("te estan saludando desde el cliente:")
    # json = json + ' desde el server'
    # emit('event', 'nani')

@socketio.on('webrtc')
def webrtc(stream):
    try:
        img_bytes = base64.b64decode(stream)        
        nparr = np.frombuffer(img_bytes, np.uint8)    
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  
        frame = get_frame_comparation(frame)
        #frame = get_face(frame)
        encoded_string = base64.b64encode(cv2.imencode('.jpg', frame[0])[1]).decode()
        labels = frame[1]

        data = {
            'img': encoded_string,
            'labels': labels
        }

        #emit('processed_webrtc', encoded_string)
        emit('processed_webrtc', data)
        
    except Exception as e:    
        # print({'result': 'errors', 'type': f"Tipo de excepción: {type(e)}", 'errors': f"Mensaje de error: {e}"})
        return jsonify({'result': 'errors', 'type': f"Tipo de excepción: {type(e)}", 'errors': f"Mensaje de error: {e}"})
    
def get_frame_comparation(frame):
    global  names_files_faces, coded_faces
    labels = []
    # Procesar la imagen con OpenCV
    frame2 = cv2.resize(frame, (0,0), None, 0.25, 0.25)    
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) #Conversion de color
    # rgb = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)    
    faces  = fr.face_locations(rgb) #identificar posibles rostros
    facescod = fr.face_encodings(rgb, faces)
    print(faces)    
    for facecod, faceloc in zip(facescod, faces) :        
        yi, xf, yf, xi = faceloc
        yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4     #Escalanos
        comparaciones = fr.compare_faces(coded_faces, facecod, 0.62) #Comparamos rostros de DB con rostro en tiempo real
        print(comparaciones)        
        simi = fr.face_distance(coded_faces, facecod)        
        min = np.argmin(simi) # Escalamos
        if comparaciones[min]:
            line_color = (0, 0, 255)              
            nombreFile = names_files_faces[min].upper()
            labels.append(nombreFile)
            cv2.putText(frame, nombreFile, (xi+6, yi-6), cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, 3)
            draw_lines(frame, xi, yi, xf, yf, dif=50, line_width=8, line_color=line_color)                    
            break
        else:             
            line_color = (0, 255, 0)
            draw_lines(frame, xi, yi, xf, yf, dif=50, line_width=8, line_color=line_color)        
    return frame, labels

def draw_lines(frame, xi, yi, xf, yf, dif,line_width, line_color):
    cv2.rectangle(img=frame, pt1=(xi, yi), pt2=(xf, yf), color=line_color, thickness=1)
    cv2.line(frame, (xi, yi), (xi+dif, yi), line_color, line_width)  # Top Left
    cv2.line(frame, (xi, yi), (xi, yi+dif), line_color, line_width)
    cv2.line(frame, (xf, yi), (xf - dif, yi), line_color, line_width)  # Top Right
    cv2.line(frame, (xf, yi), (xf, yi + dif), line_color, line_width)
    cv2.line(frame, (xi, yf), (xi + dif, yf), line_color, line_width)  # Bottom Left
    cv2.line(frame, (xi, yf), (xi, yf - dif), line_color, line_width)
    cv2.line(frame, (xf, yf), (xf - dif, yf), line_color, line_width)  # Bottom right
    cv2.line(frame, (xf, yf), (xf, yf - dif), line_color, line_width)

def get_frame_comparation1(frame):
    global  names_files_faces, coded_faces
    tiempo_inicial = float(time.time())
    # Procesar la imagen con OpenCV
    frame2 = cv2.resize(frame, (0,0), None, 0.25, 0.25)    
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) #Conversion de color
    # rgb = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)    
    faces  = fr.face_locations(rgb) #identificar posibles rostros
    facescod = fr.face_encodings(rgb, faces)
    print(f'faces: {faces}')    
    for facecod, faceloc in zip(facescod, faces) :        
        yi, xf, yf, xi = faceloc
        yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4     #Escalanos
        comparaciones = fr.compare_faces(coded_faces, facecod, 0.62) #Comparamos rostros de DB con rostro en tiempo real
        print(f'comparaciones: {comparaciones}')        
        simi = fr.face_distance(coded_faces, facecod)        
        min = np.argmin(simi) # Escalamos
        if comparaciones[min]:
            line_color = (0, 0, 255)              
            nombreFile = names_files_faces[min].upper()
            # cv2.putText(frame, nombreFile, (xi+6, yi-6), cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, 3)
            draw_lines(frame, xi, yi, xf, yf, dif=50, line_width=8, line_color=line_color)                                 
            break
        else:             
            line_color = (0, 255, 0)
            draw_lines(frame, xi, yi, xf, yf, dif=50, line_width=8, line_color=line_color)  
    print('tiempo final: ',float(time.time() - tiempo_inicial)*1000)                  
    return frame   

def get_frame_comparation2(frame):
    global  names_files_faces, coded_faces
    tiempo_inicial = float(time.time())
    # Procesar la imagen con OpenCV
    frame2 = cv2.resize(frame, (0,0), None, 0.25, 0.25)    
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) #Conversion de color
    # rgb = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)    
    faces  = fr.face_locations(rgb) #identificar posibles rostros
    facescod = fr.face_encodings(rgb, faces)
    print(f'faces: {faces}')    
    for facecod, faceloc in zip(facescod, faces) :        
        yi, xf, yf, xi = faceloc
        yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4     #Escalanos
        comparaciones = fr.compare_faces(coded_faces, facecod, 0.62) #Comparamos rostros de DB con rostro en tiempo real
        print(f'comparaciones: {comparaciones}')        
        simi = fr.face_distance(coded_faces, facecod)        
        min = np.argmin(simi) # Escalamos
        if comparaciones[min]:
            line_color = (0, 0, 255)              
            nombreFile = names_files_faces[min].upper()
            # cv2.putText(frame, nombreFile, (xi+6, yi-6), cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, 3)
            draw_lines(frame, xi, yi, xf, yf, dif=50, line_width=8, line_color=line_color)                                 
            break
        else:             
            line_color = (0, 255, 0)
            draw_lines(frame, xi, yi, xf, yf, dif=50, line_width=8, line_color=line_color)  
    print('tiempo final: ',float(time.time() - tiempo_inicial)*1000)                  
    return frame  
 
def get_face(frame):
    global names_files_faces, coded_faces
    labels = []

    # Procesar la imagen con OpenCV
    frame2 = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # Identificar posibles rostros
    faces = fr.face_locations(rgb)
    facescod = fr.face_encodings(rgb, faces)

    for facecod, faceloc in zip(facescod, faces):
        # Comparar rostros de DB con rostro en tiempo real
        comparaciones = fr.compare_faces(coded_faces, facecod, 0.62)
        simi = fr.face_distance(coded_faces, facecod)
        # BUScanos el valor mas bajo, retorna el indice
        min = np.argmin(simi)
        yi, xf, yf, xi = faceloc
        # Escalamos
        yi, xf, yf, xi = yi * 4, xf * 4, yf * 4, xi * 4
        cv2.rectangle(frame, (xi, yi), (xf, yf), (100, 500, 100), 3)

        if comparaciones[min]:
            # Dibujar rectángulo y etiqueta en el marco completo
            cv2.rectangle(frame, (xi, yi), (xf, yf), (100, 500, 100), 3)
            nombreFile = names_files_faces[min].upper()
            cv2.putText(frame, nombreFile, (xi+6, yi-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 500, 100), 2)
            labels.append(nombreFile)
            # Devolver el marco completo
            return frame
        else:
            # Devolver solo la región del rostro
            return frame#[yi:yf, xi:xf]
    # Devolver el marco completo si no se detectaron rostros
    return frame, labels
   
#buscar  faces of db---------------------------------------------------------
@socketio.on('buscarFaces')
def buscar_faces(stream):
    global  images_face, names_files_faces, coded_faces
    
    img_bytes = base64.b64decode(stream.split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    # nparr = np.fromstring(img_bytes, np.uint8)    
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)    
    
    # # Procesar la imagen con OpenCV
    frame2 = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    #Conversion de color
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    # rgb = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    #BUScano5 los rostros
    faces  = fr.face_locations(rgb)
    facescod = fr.face_encodings(rgb, faces)
    print(faces)
    comp1 = 100
    # Iteranos
    for facecod, faceloc in zip(facescod, faces) :
        #Comparamos rostros de DB con rostro en tiempo real
        comparacion = fr.compare_faces(coded_faces, facecod, 0.62)
        print(comparacion)
        #Calculamos la solicitud
        simi = fr.face_distance(coded_faces, facecod)
        # print(simi)

        #BUScanos el valor mas bajo, retorna el indice
        min = np.argmin(simi)
        print(comp1)
        if comparacion[min]:
            # print('nomre: ', nombre)
            #EXtraenos coordenadas
            yi, xf, yf, xi = faceloc
            #Escalanos
            yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4
            indice = comparacion.index(True)
            # print('encontro------: ', nombre)

            # Comparanos
            if comp1 != indice:
                nombre = names_files_faces[min].upper()
                
                #Para dibujar canbianos colores
                # r = random.randrange(0, 255, 50)
                # g = random.randrange(0, 255, 50)
                # b = random.randrange(0, 255, 50)
                comp1 = indice
                # print('comp1 == indice:')
                cv2.rectangle(frame, (xi, yi), (xf, yf), (100, 500, 100), 3)                           
                # cv2.rectangle(frame, (xi, yi), (xf, yf), (r, g, b), 3)                           
                cv2.putText(frame, nombre, (xi+6, yi-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 500, 100), 2)
                break
                    
    encoded_string = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()          
    # _, buffer = cv2.imencode('.jpg', img)
    # edges_b64 = base64.b64encode(buffer)
    # # Convertir la cadena de bytes a una cadena de texto
    # encoded_string = edges_b64.decode('utf-8')
    # # print(stream)
    print('fin de server')
    emit('processed_buscar_faces', encoded_string)
    
    
#reconocoe faces----------------------------------------------------
@socketio.on('stream')
def stream(stream):
    img_bytes = base64.b64decode(stream.split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # # Procesar la imagen con OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Definir el color y el grosor del rectángulo
    color = (0, 0, 255) # Rojo
    grosor = 3
    for (x, y, w, h) in faces:
        # Definir punto de inicio y punto final para el rectángulo
        punto_inicial = (x, y-65)
        punto_final = (x + w, y+h + 20)
        # print(punto_inicial, ' : ',punto_final)
        cv2.rectangle(img, punto_inicial, punto_final, color, grosor)  
    encoded_string = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()         
    #similar  como la linea de arriba, solo mas detallado
    # _, buffer = cv2.imencode('.jpg', img)
    # edges_b64 = base64.b64encode(buffer)
    # # Convertir la cadena de bytes a una cadena de texto
    # encoded_string = edges_b64.decode('utf-8')
    # # print(stream)
    emit('processed_stream', encoded_string)

#comparte imagen 
# @socketio.on('stream')
# def stream(stream):
#     img_bytes = base64.b64decode(stream.split(',')[1])
#     nparr = np.frombuffer(img_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     # # Procesar la imagen con OpenCV
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 100, 200)
#     # # Codificar la imagen procesada en base64
#     _, buffer = cv2.imencode('.jpg', edges)
#     edges_b64 = base64.b64encode(buffer)
#     print(stream)
#     emit('processed_stream', edges_b64.decode('utf-8'))


@socketio.on("image")
def receive_image(image):
    # Decode the base64-encoded image data
    image = base64_to_image(image)

    # Perform image processing using OpenCV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(gray, (640, 360))

    # Encode the processed image as a JPEG-encoded base64 string
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, frame_encoded = cv2.imencode(".jpg", frame_resized, encode_param)
    processed_img_data = base64.b64encode(frame_encoded).decode()

    # Prepend the base64-encoded string with the data URL prefix
    b64_src = "data:image/jpg;base64,"
    processed_img_data = b64_src + processed_img_data

    # Send the processed image back to the client
    emit("processed_image", processed_img_data)
    
    
def base64_to_image(base64_string):
    # Extract the base64 encoded binary data from the input string
    base64_data = base64_string.split(",")[1]
    # Decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_data)
    # Convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode the numpy array as an image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


# función para procesar la imagen
def process_image(image_data):
    # convertir la imagen de bytes a Image de Pillow
    print('hola')
    img = Image.open(BytesIO(image_data))
    
    # hacer algo con la imagen, por ejemplo, redimensionar
    resized_img = img.resize((640, 480))
    # convertir la imagen de Pillow a bytes para enviarla de vuelta al cliente
    buffer = BytesIO()
    resized_img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    
    return img_bytes


    
#iniciar el hilo: thread    
def start_video_thread():
    global hilos, dim_hilos, thread0, thread1, thread2, thread3
    # global queue, sw_hilos
    # saved_files1(dir=ENV.DIR_FACES)
    # for i in range(0, dim_hilos): 
    #     if hilos[i]  and (hilos[i] .is_alive()) :
    #         print(f'el HILO:{i} ya se esta ejecutando---------------------')  
    #     else:
    #         hilos[i]  = Hilo(i, daemon=True, emit=f'processed_webrtc{i}')
    #         hilos[i].start();         
    # p1 = multiprocessing.Process(target=pru, args=(queue,))
    # p1.start()
    # saved_files1(dir=ENV.DIR_FACES)    
    if thread0  and (thread0 .is_alive()) :
        print(f'el HILO: thread0 ya se esta ejecutando---------------------')  
    else:
        # thread0 = Thread(target=prueba_hilo) 
        thread0 = Thread(target=send_video0) 
        thread0.daemon = True
        thread0.start();   
        
    if thread1  and (thread1 .is_alive()) :
        print(f'el HILO: thread1 ya se esta ejecutando---------------------')  
    else:
        thread1  = Thread(target=send_video1) 
        thread1.daemon = True
        thread1.start();  
                   
       
    if thread2  and (thread2.is_alive()) :
        print(f'el HILO: thread2 ya se esta ejecutando---------------------')  
    else:
        # thread2 = Thread(target=prueba_hilo) 
        thread2 = Thread(target=send_video2) 
        thread2.daemon = True
        thread2.start();   
        
    if thread3  and (thread3.is_alive()) :
        print(f'el HILO: thread3 ya se esta ejecutando---------------------')  
    else:
        # thread3 = Thread(target=prueba_hilo) 
        thread3 = Thread(target=send_video3) 
        thread3.daemon = True
        thread3.start();   
       
    # p1 = multiprocessing.Process(target=pru)
    # with ThreadPoolExecutor() as executor:j
    #     executor.submit(send_video0)
    #     executor.submit(send_video1)
        
    

def pru(queue):    
    capture  = cv2.VideoCapture(0)
    i = 0
    while capture.isOpened():
    # while True:
        ret, frame = capture.read() # lee un fotograma de la cámara      
        if ((not ret) or (i+1 >= 256)): break        
        # time.sleep(1/4)
        tiempo_inicial = float(time.time())
        #---------------------------------------------------------------                     
        draw_compare_faces(frame)                        
        #---------------------------------------------------------------       
        tiempo_final = round((time.time()-tiempo_inicial)*1000, 2); 
        print(f'--TIEMPO pru: {tiempo_final}')                      
        # encoded_string = base64.b64encode(cv2.imencode('.jpg', frame[0])[1]).decode()   
        encoded_string = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()       
        queue.put(encoded_string)    
        i+=1        
    capture.release() # libera la cámara
    print('finaliso pru --------------------------')
        
def prueba_hilo():
    global queue, clients    
    # time.sleep(30)    
    while True:
        if ((clients == 0)): break 
        if  queue.qsize() > 0 :
            data = {
                    'img': queue.get(),
                    'labels': '',
                    'id': 0
                }
            socketio.emit('processed_webrtc', data)
            # print('--prueba_hilo')
    print('finaliso prueba_hilo --------------------------')
    

def buscar_pos(array, elem):
    try:        
        return array.index(elem)
    except ValueError:
        return -1  # Si no se encuentra ningún True en el array   

    
def send_video0():
    global clients, sw_hilos
    # Q = deque(maxlen=128)
    while sw_hilos and (clients != 0):
        capture  = cv2.VideoCapture(0) # selecciona la cámara 0 como fuente de video
        ultimo_tiempo = time.time()   # Tiempo inicial
        while capture.isOpened():
            ret, frame = capture.read() # lee un fotograma de la cámara
            if ((not ret) or (clients == 0) or (not sw_hilos)): break        
            # time.sleep(1/2)
            #---------------------------------------------------------------   
            tiempo_actual = (time.time())   
            labels = []
            time.sleep(1/16)        
            if ( (tiempo_actual-ultimo_tiempo) >= 3): 
                ultimo_tiempo = tiempo_actual
                print(f'paso 2 seg------{tiempo_actual}')    
                # labels = draw_compare_faces(frame)                                    
            tiempo_final = round((time.time()-tiempo_actual)*1000, 2); 
            print('--TIEMPO FINAL0: ', tiempo_final)               
            # Q.append(tiempo_final)        
            #---------------------------------------------------------------                     
            encoded_string = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()       
            data = {
                'img': encoded_string,
                'labels': labels,
                'id': 0
            }
            socketio.emit('processed_webrtc', data)
            # socketio.emit('event', data)
        print('finaliso send_video0 --------------------------')
        capture.release() # libera la cámara
        cv2.destroyAllWindows()
    # print(Q)
    # promedio = np.array(Q).mean(axis=0)   #realizar un promedio de predicción sobre el historial de predicciones anteriores
    # print("PROMEDIO: ", promedio)
    
    
def send_video1():
    global clients, sw_hilos
    
    while sw_hilos and (clients != 0):
        capture  = cv2.VideoCapture(Video.VIDEO1) # selecciona la cámara 0 como fuente de video
        # capture  = cv2.VideoCapture('G:\materias\cursos\python\descargados\Violence-Alert-System\Violence-Detection\Testing-videos\V_19.mp4') # selecciona la cámara 0 como fuente de video
        ultimo_tiempo = time.time()   # Tiempo inicial
        while capture.isOpened():
            ret, frame = capture.read() # lee un fotograma de la cámara      
            if ((not ret) or (clients == 0) or (not sw_hilos)): break        
            time.sleep(1/10)
            tiempo_actual = float(time.time())
            # if ((System.currentTimeMillis()-time)% delay == 0){
            #---------------------------------------------------------------                     
            if ((tiempo_actual-ultimo_tiempo) >= 1.5): 
                ultimo_tiempo = tiempo_actual
                # print(f'paso 2 seg------{tiempo_actual}')    
                draw_detect_faces(frame)                        
            #---------------------------------------------------------------       
            tiempo_final = round((time.time()-tiempo_actual)*1000, 2); 
            print(f'--TIEMPO FINAL1: {tiempo_final}' )                      
            # encoded_string = base64.b64encode(cv2.imencode('.jpg', frame[0])[1]).decode()   
            encoded_string = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()                   
            data = {
                'img': encoded_string,
                'labels': [],
                'id': 1
            }
            socketio.emit('processed_webrtc',data)
            # socketio.emit('event', data)
        print('finaliso send_video1 --------------------------')
        capture.release() # libera la cámara
        cv2.destroyAllWindows()
        
    
def send_video2():
    global clients     
    
    model = load_model('modelnew.h5')
    while sw_hilos and (clients != 0):
        capture  = cv2.VideoCapture(Video.VIDEO2) # selecciona la cámara 0 como fuente de video
        ultimo_tiempo = time.time()   # Tiempo inicial
        
        while capture.isOpened():
            ret, frame = capture.read() # lee un fotograma de la cámara      
            if ((not ret) or (clients == 0) or (not sw_hilos)): break        
            time.sleep(1/10)
            #---------------------------------------------------------------                     
            tiempo_actual = float(time.time())
            sw = False
            if ((tiempo_actual-ultimo_tiempo) >= 1.5): 
                ultimo_tiempo = tiempo_actual
                sw = draw_violence_detection(frame=frame, model=model)                        
            tiempo_final = round((time.time()-tiempo_actual)*1000, 2); 
            print(f'--TIEMPO FINAL2: {tiempo_final}' )                      
            #---------------------------------------------------------------       
            encoded_string = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()                   
            data = {
                'img': encoded_string,
                'labels': '',
                'sw': 1 if sw else 0,
                'id': 2
            }
            socketio.emit('processed_webrtc', data)
            # socketio.emit('event', data)
        print('finaliso send_video2 --------------------------')
        capture.release() # libera la cámara
        cv2.destroyAllWindows()
        
    
def send_video3():
    global clients 
    while sw_hilos and (clients != 0):
        capture  = cv2.VideoCapture(Video.VIDEO3) # selecciona la cámara 0 como fuente de video
        ultimo_tiempo = time.time()   # Tiempo inicial
        model = load_model('modelnew.h5')
        
        while capture.isOpened():
            ret, frame = capture.read() # lee un fotograma de la cámara      
            if ((not ret) or (clients == 0) or (not sw_hilos)): break        
            time.sleep(1/10)
            #---------------------------------------------------------------                     
            tiempo_actual = float(time.time())
            sw = False
            if ((tiempo_actual-ultimo_tiempo) >= 2.5): 
                ultimo_tiempo = tiempo_actual
                sw = draw_violence_detection(frame=frame, model=model)                        
            tiempo_final = round((time.time()-tiempo_actual)*1000, 2); 
            print(f'--TIEMPO FINAL3: {tiempo_final}' )     
                            
            #---------------------------------------------------------------       
            encoded_string = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()                   
            data = {
                'img': encoded_string,
                'labels': '',
                'sw': 1 if sw else 0,
                'id': 3
            }
            socketio.emit('processed_webrtc', data)
            # socketio.emit('event', data)
        print('finaliso send_video3 --------------------------')
        capture.release() # libera la cámara
        cv2.destroyAllWindows()
        
    
    
def draw_detect_faces(frame):
    ubicaciones, confianzas = cv.detect_face(frame)
    for (x, y, w, h), confianza in zip(ubicaciones, confianzas):
        cv2.rectangle(frame, (x, y), (w, h), Color.VERDE, 2)                    
        cv2.putText(frame, f"Confianza: {confianza:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Color.VERDE, 2)        
        
        
def draw_compare_faces(frame):
    frame2 = cv2.resize(frame, (0,0), None, 1/4, 1/4)                              
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) #Correccion de color
    ubicaciones_caras  = fr.face_locations(rgb) #identificar posibles rostros                      
    # print(f'--------------ubicaciones_caras: {ubicaciones_caras}') 
    caras_desconocidas = fr.face_encodings(rgb, ubicaciones_caras)#identificar ubicaciones de rostros
    # print(f'--------------caras_desconocidas: {caras_desconocidas}')
    sw = False if len(caras_desconocidas) == 0 else  True
    labels = []
    for facecod, faceloc in zip(caras_desconocidas, ubicaciones_caras) :     #for tarda 0.5 ms, por cada vuelta                           
        yi, xf, yf, xi = tuple(val * 4 for val in faceloc) # aca valor de la tupla = val * 4
        comparaciones = fr.compare_faces(coded_faces, facecod, 0.62) #Comparamos rostros de DB con rostro en tiempo real
        # print(f'comparaciones: {comparaciones}')                    
        i = buscar_pos(comparaciones, True)                 
        if i == -1: # encontrar el índice del primer elemento True                          
            draw_lines(frame, xi, yi, xf, yf, dif=50, line_width=8, line_color=(0, 255, 0) )
        else:             
            draw_lines(frame, xi, yi, xf, yf, dif=50, line_width=8, line_color=(0, 0, 255) ) 
            nombreFile = names_files_faces[i].upper()               
            labels.append(nombreFile)            
            cv2.putText(frame, nombreFile, (xi+6, yi-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)                                                                          
    #         # break       
    return labels


def draw_violence_detection(frame, model):
    frame_copy = frame.copy()
    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB) #convertir la imagen de BGR (el formato utilizado por OpenCV) a RGB
    frame_copy = cv2.resize(frame_copy, (128, 128)).astype("float32")#imagen a 128 x 128 píxeles
    frame_copy = frame_copy.reshape(128, 128, 3) / 255 #se divide cada valor de píxel por 255 para normalizar los valores de los píxeles a un rango entre 0 y 1.                
    preds = model.predict(np.expand_dims(frame_copy, axis=0))[0] #haga predicciones en el marco
    # print("preds",preds)
    bool = (preds > 0.60)[0]                      
    if (bool):
        text_color = Color.ROJO                                       
    else:  
        text_color = Color.VERDE
    cv2.putText(
        img=frame,
        text="Violence: {}".format(bool),
        org=(35, 50),
        fontFace= cv2.FONT_HERSHEY_SIMPLEX ,
        fontScale=1.25,
        color=text_color,
        thickness=3
    ) 
    return bool