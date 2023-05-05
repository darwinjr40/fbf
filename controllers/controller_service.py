from flask import Flask, Blueprint, render_template, request, jsonify, url_for
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

service_bp = Blueprint('service', __name__)

#service socket ------------------------------------------------
# Definir la vista "inicio"
@service_bp.route('/')
def inicio():
    return render_template('prueba22.html')

@service_bp.route('/buscar')
def buscar():
    return render_template('buscar.html')

@service_bp.route('/webrtc')
def webrtc():
    return render_template('webrtc.html')

#services ------------------------------------------------
@service_bp.route('/example')
def example():
    # return render_template('inicio.html')
    return 'nise123'

@service_bp.route('/process_image', methods=['POST'])
def process_image():
    try:
#     # Código que puede lanzar una excepción
        # Obtener la imagen de la solicitud POST
        image_file = request.files['image'].read()
        # nombre = request.form.get("nombre")
        
        # decodificar datos de la imagen en un arreglo de bytes
        nparr = np.fromstring(image_file, np.uint8)
        
        # decodificar arreglo de bytes en matriz numpy
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convertir la imagen a una matriz numpy usando OpenCV
        # image_array = cv2.imdecode(np.frombuffer(image_file, np.uint8), cv2.IMREAD_COLOR)
        
        # Procesar la imagen con OpenCV
        # Aquí se puede agregar cualquier código de procesamiento de imagen que se desee
        xi, yi, xf, yf = 1, 1, 100, 100
        r = random.randrange(0, 255, 50)
        g = random.randrange(0, 255, 50)
        b = random.randrange(0, 255, 50)
        
        cv2.rectangle(img_array, (xi, yi), (xf, yf), (r, g, b), 3)
        
        # Codificar la imagen en formato jpg
        ret, buffer = cv2.imencode('.jpg', img_array)
        # return 'nise'
        image_64_encode = base64.b64encode(buffer)
        return image_64_encode.decode('utf-8')
        return jsonify({'result': 'success', 'processed_image': image_64_encode.decode('utf-8')})  
    # except ValueError:
    except Exception as e:
        return jsonify({'result': 'errors', 'type': f"Tipo de excepción: {type(e)}", 'errors': f"Mensaje de error: {e}"})  
    



@service_bp.route('/process', methods=['POST'])
def process():
    try:
        # # Accedemos a la carpeta
        path = 'personal'
        images = []
        clases = []
        lista = os.listdir(path)
        # lista = os.listdir('.')
        print('---------------------------------------------------------------')
        print(lista)


        # Variables
        comp1 = 100
        # #Leemos los rostros del DB
        for lis in lista:
            #Leemos Las imagenes de los rostros
            imgdb = cv2.imread(f'{path}/{lis}')
            # ALmacenamos imagen
            images.append(imgdb)
            #ALmacenamos nombre
            clases.append(os.path.splitext(lis)[0])
        # print (clases)
        
        # #Llamanos la funcion
        rostroscod = codrostros(images)
        print(rostroscod)
        #Leemos los fotogramas
        
        # Obtener la imagen de la solicitud POST
        image_file = request.files['image'].read()
        # nombre = request.form.get("nombre")
        
        # decodificar datos de la imagen en un arreglo de bytes
        nparr = np.fromstring(image_file, np.uint8)
        
        # decodificar arreglo de bytes en matriz numpy
        # img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_array = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
        
        frame = img_array
        #Reducimos las imagenes para mejor procesamiento
        # frame2 = cv2.resize(frame, (0,0), None, 1, 1)
        frame2 = cv2.resize(frame, (0,0), None, 0.25, 0.25)
        #Conversion de color
        rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        # rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        #BUScano5 los rostros
        faces  = fr.face_locations(rgb)
        # print(len(faces))
        # return str(len(faces))
        # return  jsonify({'result': len(faces)})  
        facescod = fr.face_encodings(rgb, faces)
        # print(facescod)
        # Iteranos
        for facecod, faceloc in zip(facescod, faces) :
            #Comparamos rostros de DB con rostro en tiempo real
            comparacion = fr.compare_faces(rostroscod, facecod, 0.7)
            print(comparacion)
            # if comparacion[0]:
                # return 'nise'
            #Calculamos la solicitud
            simi = fr.face_distance(rostroscod, facecod)
            # print(simi)

            #BUScanos el valor mas bajo, retorna el indice
            min = np.argmin(simi)
            
            if comparacion[min]:
                nombre = clases[min].upper()
                print(nombre)
                #EXtraenos coordenadas
                print(faceloc)
                yi, xf, yf, xi = faceloc
                #Escalanos
                # yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4
                yi, xf, yf, xi = yi, xf*2, yf*2, xi
                indice = comparacion.index(True)
# https://meet.google.com/iuq-ryii-uze
                # Comparanos
                if comp1 != indice:
                    #Para dibujar canbianos colores
                    r = random.randrange(0, 255, 50)
                    g = random.randrange(0, 255, 50)
                    b = random.randrange(0, 255, 50)
                    comp1 = indice

                if comp1 == indice:
                    #dibujamos                    
                    cv2.rectangle(frame, (xi, yi), (xf, yf), (r, g, b), 3)
                    # cv2.rectangle(frame, (xi, yi), (xf, yf-35), (r, g, b), cv2.FILLED)
                    cv2.putText(frame, nombre, (xi+6, yi-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    # horario(nombre)    
        
        # Mostranos_ Franes
        # cv2.imshow("Reconocimiento Facial", frame)
        
        # Codificar la imagen en formato jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        image_64_encode = base64.b64encode(buffer)
        return image_64_encode.decode('utf-8')

        return jsonify({'result': 'success'})  
    
    except Exception as e:    
        return jsonify({'result': 'errors', 'type': f"Tipo de excepción: {type(e)}", 'errors': f"Mensaje de error: {e}"})  
    
    
# @example_bp.route('/process', methods=['POST'])
# def process():
#     try:
#         return jsonify({'result': 'success'})  








        

        # #Realizanos VidepCaptura
        # cap = cv2.VideoCapture(0)

        # #Empezano0s
        # while True:
        #     #Leemos los fotogramas
        #     ret, frame = cap.read()
        #     #Reducimos las imagenes para mejor procesamiento
        #     frame2 = cv2.resize(frame, (0,0), None, 0.25, 0.25)
        #     #Conversion de color
        #     rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        #     # rgb = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        #     #BUScano5 los rostros
        #     faces  = fr.face_locations(rgb)
        #     facescod = fr.face_encodings(rgb, faces)

        #     # Iteranos
        #     for facecod, faceloc in zip(facescod, faces) :
        #         #Comparamos rostros de DB con rostro en tiempo real
        #         comparacion = fr.compare_faces(rostroscod, facecod)

        #         #Calculamos la solicitud
        #         simi = fr.face_distance(rostroscod, facecod)
        #         # print(simi)

        #         #BUScanos el valor mas bajo, retorna el indice
        #         min = np.argmin(simi)
                
        #         if comparacion[min]:
        #             nombre = clases[min].upper()
        #             print(nombre)
        #             #EXtraenos coordenadas
        #             yi, xf, yf, xi = faceloc
        #             #Escalanos
        #             yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4
        #             indice = comparacion.index(True)

        #             # Comparanos
        #             if comp1 != indice:
        #                 #Para dibujar canbianos colores
        #                 r = random.randrange(0, 255, 50)
        #                 g = random.randrange(0, 255, 50)
        #                 b = random.randrange(0, 255, 50)
        #                 comp1 = indice

        #             if comp1 == indice:
        #                 #dibujamos
                        
        #                 cv2.rectangle(frame, (xi, yi), (xf, yf), (r, g, b), 3)
        #                 # cv2.rectangle(frame, (xi, yi), (xf, yf-35), (r, g, b), cv2.FILLED)
        #                 cv2.putText(frame, nombre, (xi+6, yi-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #                 # horario(nombre)    
            
        #     # Mostranos_ Franes
        #     cv2.imshow("Reconocimiento Facial", frame)
            
        #     #Leenos el teclado
        #     # t = cv2.waitKey(5)
        #     t = cv2.waitKey(30)
        #     if t == 27:
        #         break

        # cv2.destroyAllWindows()
        # cap.release()  
    #     return 'bien'
    # except Exception as e:                
    #     return 'error'





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

#Hora de ingreso
def horario (nombre):
    #Abrimos el archivo en modo 1ectura y escnitura
    with open('Horario.csv','r+') as h:
        # Leenos la infornacion
        data = h.readline()
        #Creamos lista de nonbres
        listanombres = []
        #Iteranos cada linea del doc
        for line in data:
            #Buscamos la entrada y la diferencianos con,
            entrada = line.splitC(',')
            #ALmacenamos los nonbres
            listanombres.append(entrada[0])
            
        #Venificamos si ya hemos alnacenado el nombre
        if nombre not in listanombres :
            #Extraenos informacion actual
            info = datetime.now()
            #EXtraemos fecha
            fecha = info.strftime('%Y:%m:%d')
            #Extraemos hora
            hora = info.strftime('%H:%M:%S')

            #Guardanos la informacion
            h.writelines (f'\n{nombre:}, {fecha} , {hora}')
            print(info)

  