from flask import Flask, Blueprint, render_template, request, jsonify, url_for
import  os, cv2, base64, random, numpy as np, face_recognition as fr
from werkzeug.utils import redirect
from werkzeug.exceptions import abort
from flask_socketio import SocketIO, emit
from io import BytesIO
from PIL import Image
from threading import Thread
import time
# from app import socketio
#-----------------------------------------------------
def init():
    # # Accedemos a la carpeta
    global clases, images
    path = 'personal'    
    lista = os.listdir(path)
    # lista = os.listdir('.')
    print(lista)    
    # #Leemos los rostros del DB
    for lis in lista:
        #Leemos Las imagenes de los rostros
        imgdb = cv2.imread(f'{path}/{lis}')
        # ALmacenamos imagen
        images.append(imgdb)
        #ALmacenamos nombre
        clases.append(os.path.splitext(lis)[0])
    print (clases)
    
    
    


#-----------------------------------------------------
#Funcion para codificar los rostros
def codrostros(images):
    listacod = []
    # Iterano0s
    for img in images:
        # Correccion de color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #Codificamos la imagen
        cod = fr.face_encodings(img)[0]
        #ALmacenamos
        listacod.append(cod)

    return listacod


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
images = []
clases = []
init()
rostroscod = codrostros(images)
print('xd')
print(rostroscod)
clients = 0
thread = None


#socketio = SocketIO(None, cors_allowed_origins="*", logger=True, engineio_logger=True, ping_timeout=300)
socketio = SocketIO(None, cors_allowed_origins="*")


#get instancia------------------------------------------------------
def create_socketio_app(app):
    socketio.init_app(app)
    return socketio
#socket default------------------------------------------------------
@socketio.on('connect')
def handle_connect():
    global clients, thread
    clients = len(socketio.server.manager.rooms['/'].keys()) - 1  # Restar 1 para excluir al propio cliente
    print(f"Número de clientes conectados: {clients}")
    print('se conectaron')
    if  (not thread) or  (not thread.is_alive()):
        start_video_thread()
    else :
        print('el hilo ya se esta ejecutando---------------------')    
    # emit('processed_webrtc', 'asdasdsd')
    
@socketio.on('disconnect')
def handle_connect():
    global clients
    print('se desconectaron')
    clients -=1
    print(f"Número de clientes conectados: {clients}")
    # emit('processed_webrtc', 'asdasdsd')
    # start_repeating_task()
    

#socket eventos---------------------------------------------------------
@socketio.on('event')
def event(json):
    print("te estan saludando desde el cliente:")
    # json = json + ' desde el server'
    emit('event', 'nani')

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
    global  clases, rostroscod
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
        comparaciones = fr.compare_faces(rostroscod, facecod, 0.62) #Comparamos rostros de DB con rostro en tiempo real
        print(comparaciones)        
        simi = fr.face_distance(rostroscod, facecod)        
        min = np.argmin(simi) # Escalamos
        if comparaciones[min]:
            line_color = (0, 0, 255)              
            nombreFile = clases[min].upper()
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
    
def get_face(frame):
    global clases, rostroscod
    labels = []

    # Procesar la imagen con OpenCV
    frame2 = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # Identificar posibles rostros
    faces = fr.face_locations(rgb)
    facescod = fr.face_encodings(rgb, faces)

    for facecod, faceloc in zip(facescod, faces):
        # Comparar rostros de DB con rostro en tiempo real
        comparaciones = fr.compare_faces(rostroscod, facecod, 0.62)
        simi = fr.face_distance(rostroscod, facecod)
        # BUScanos el valor mas bajo, retorna el indice
        min = np.argmin(simi)
        yi, xf, yf, xi = faceloc
        # Escalamos
        yi, xf, yf, xi = yi * 4, xf * 4, yf * 4, xi * 4
        cv2.rectangle(frame, (xi, yi), (xf, yf), (100, 500, 100), 3)

        if comparaciones[min]:
            # Dibujar rectángulo y etiqueta en el marco completo
            cv2.rectangle(frame, (xi, yi), (xf, yf), (100, 500, 100), 3)
            nombreFile = clases[min].upper()
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
    global  images, clases, rostroscod
    
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
        comparacion = fr.compare_faces(rostroscod, facecod, 0.62)
        print(comparacion)
        #Calculamos la solicitud
        simi = fr.face_distance(rostroscod, facecod)
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
                nombre = clases[min].upper()
                
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


    
    
def start_video_thread():
    global thread
    print('todo bien')
    thread = Thread(target=send_video)
    thread.daemon = True
    thread.start()
    
    
def send_video():
    global clients
    capture  = cv2.VideoCapture(0) # selecciona la cámara 0 como fuente de video
    while capture.isOpened():
        ret, frame = capture.read() # lee un fotograma de la cámara      
        if ((not ret) or (clients == 0)): break
        
        # time.sleep(0.05)
        frame = get_frame_comparation(frame)                        
        encoded_string = base64.b64encode(cv2.imencode('.jpg', frame[0])[1]).decode()       
        data = {
            'img': encoded_string,
            'labels': ''
        }
        socketio.emit('processed_webrtc', data)
    print('finaliso send_video --------------------------')
    capture.release() # libera la cámara