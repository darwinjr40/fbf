# Importaciones de librerías estándar de Python
import base64
import threading
import time
import cv2

# Importaciones de módulos locales
import controllers.controller_socket as controller_socket

class Hilo(threading.Thread): 
    
    def __init__(self, id, daemon, emit, target=None):
        super().__init__(target=target)
        self.daemon = daemon;
        self.id = id
        self.emit = emit
        
    def run(self):
        super().run()
        # self._send_video0()
        print(f'-----------FINALIZÓ EL HILO: {self.id}--------------------------')


    def _send_video0(self):
        capture  = cv2.VideoCapture(self.id) # selecciona la cámara 0 como fuente de video
        while capture.isOpened():
            ret, frame = capture.read() # lee un fotograma de la cámara      
            if ((not ret) or (controller_socket.clients == 0)): break        
            time.sleep(1/8) #2 fotograma por segundo = 1/2
            frame = controller_socket.get_frame_comparation(frame)                        
            encoded_string = base64.b64encode(cv2.imencode('.jpg', frame[0])[1]).decode()       
            data = {
                'img': encoded_string,
                'labels': '',
                'id': self.id
            }
            controller_socket.socketio.emit('processed_webrtc', data)
            # controller_socket.socketio.emit(self.emit, data)
        capture.release() # libera la cámara