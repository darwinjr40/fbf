from flask import Flask, Blueprint, render_template, request, jsonify, url_for
import  os, cv2, base64, random, numpy as np, face_recognition as fr
from werkzeug.utils import redirect
from werkzeug.exceptions import abort
from flask_socketio import SocketIO, emit
from io import BytesIO
from PIL import Image



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


socketio = SocketIO()


#get instancia
def create_socketio_app(app):
    socketio.init_app(app)
    return socketio


#socket---------------------------------------------------------
@socketio.on('event')
def event(json):
    print("te estan saludando desde el cliente:" + json)
    json = json + ' desde el server'
    emit('event',json)
#webrtc------------------------------
@socketio.on('webrtc')
def webrtc(stream):
    try:
        img_bytes = base64.b64decode(stream)        
        nparr = np.frombuffer(img_bytes, np.uint8)    
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  
        frame = get_frame_comparation(frame)
        encoded_string = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()          
        emit('processed_webrtc', encoded_string)
        
    except Exception as e:    
        print({'result': 'errors', 'type': f"Tipo de excepción: {type(e)}", 'errors': f"Mensaje de error: {e}"})
        # return jsonify({'result': 'errors', 'type': f"Tipo de excepción: {type(e)}", 'errors': f"Mensaje de error: {e}"})  
    
def get_frame_comparation(frame):
    global  clases, rostroscod
    # Procesar la imagen con OpenCV
    frame2 = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    #Conversion de color
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    # rgb = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    #identificar posibles rostros
    faces  = fr.face_locations(rgb)
    facescod = fr.face_encodings(rgb, faces)
    print(faces)    
    for facecod, faceloc in zip(facescod, faces) :
        #Comparamos rostros de DB con rostro en tiempo real
        comparaciones = fr.compare_faces(rostroscod, facecod, 0.62)
        print(comparaciones)        
        simi = fr.face_distance(rostroscod, facecod)        
        #BUScanos el valor mas bajo, retorna el indice
        min = np.argmin(simi)        
        if comparaciones[min]:
            yi, xf, yf, xi = faceloc
            #Escalanos
            yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4                                
            cv2.rectangle(frame, (xi, yi), (xf, yf), (100, 500, 100), 3)                                       
            nombreFile = clases[min].upper()            
            cv2.putText(frame, nombreFile, (xi+6, yi-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 500, 100), 2)
            break
    return frame 
   
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


