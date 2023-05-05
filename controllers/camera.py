import cv2
# import face_recognition as fr   #pip install face_recognition
import face_recognition as fr
import os
import random
import numpy as np
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class VideoCamera(object) :
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        print('xdxd---------------------')
       
        
    def __del__(self):
        self.video.releast()
        
    def get_frame(self):
        ret, frame = self.video.read()
        # Codifica el frame en formato JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
    
    def  get_detect_faces(self):
        ret, frame = self.video.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            break
        # Codifica el frame en formato JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
    #--------------------------------------------------------------------------------
    def get_compare_faces(self):
                # Accedemos a la carpeta
        path = 'personal'
        images = []
        clases = []
        lista = os.listdir(path)
        # lista = os.listdir('.')
        print(lista)


        # # Variables
        comp1 = 100
        # Leemos los rostros del DB
        for lis in lista:
            #Leemos Las imagenes de los rostros
            imgdb = cv2.imread(f'{path}/{lis}')
            # ALmacenamos imagen
            images.append(imgdb)
            #ALmacenamos nombre
            clases.append(os.path.splitext(lis)[0])
        print (clases)

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
        #Llamanos la funcion        
        rostroscod = codrostros(images)
        # print(rostroscod)
        
        

        #Empezano0s
        while True:
            #Leemos los fotogramas
            ret, frame = self.video.read()
            #Reducimos las imagenes para mejor procesamiento
            # frame2 = cv2.resize(frame, (0,0), None, 0.25, 0.25)
            # #Conversion de color
            # rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            # # rgb = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

            # #BUScano5 los rostros
            # faces  = fr.face_locations(rgb)
            # facescod = fr.face_encodings(rgb, faces)

            # # Iteranos
            # for facecod, faceloc in zip(facescod, faces) :
            #     #Comparamos rostros de DB con rostro en tiempo real
            #     comparacion = fr.compare_faces(rostroscod, facecod, 0.7)

            #     #Calculamos la solicitud
            #     simi = fr.face_distance(rostroscod, facecod)
            #     # print(simi)

            #     #BUScanos el valor mas bajo, retorna el indice
            #     min = np.argmin(simi)
                
            #     if comparacion[min]:
            #         nombre = clases[min].upper()
            #         print(nombre)
            #         #EXtraenos coordenadas
            #         yi, xf, yf, xi = faceloc
            #         #Escalanos
            #         yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4
            #         indice = comparacion.index(True)

            #         # Comparanos
            #         if comp1 != indice:
            #             #Para dibujar canbianos colores
            #             r = random.randrange(0, 255, 50)
            #             g = random.randrange(0, 255, 50)
            #             b = random.randrange(0, 255, 50)
            #             comp1 = indice

            #         if comp1 == indice:
            #             #dibujamos
                        
            #             cv2.rectangle(frame, (xi, yi), (xf, yf), (r, g, b), 3)
            #             # cv2.rectangle(frame, (xi, yi), (xf, yf-35), (r, g, b), cv2.FILLED)
            #             cv2.putText(frame, nombre, (xi+6, yi-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            #             # horario(nombre)    
            
            # Mostranos_ Franes
            # cv2.imshow("Reconocimiento Facial", frame)
            # Codifica el frame en formato JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
           
    
class Singleton:
    __instance = None

    def __init__(self):
        if Singleton.__instance is  None:            
            Singleton.__instance = VideoCamera()

    @staticmethod
    def get_instance():
        if Singleton.__instance is None:
            Singleton()
        return Singleton.__instance    