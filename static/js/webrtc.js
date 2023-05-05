const socket = io();
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startButton = document.getElementById('start-button');
const ctx = canvas.getContext("2d");
const img = document.getElementById("imagen");

startButton.addEventListener('click', () => {
    navigator.mediaDevices.getUserMedia({video: true})
        .then(stream => {
          video.srcObject = stream;
          // video.play();          
          setInterval(() => {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const base64ImageData = canvas.toDataURL('image/png', 0.5).split(',')[1];
            socket.emit('webrtc', base64ImageData);
          }, 500);
        })
        .catch(error => {
            console.log('Error accessing camera:', error.message);
        });
});

video.addEventListener("loadedmetadata", () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  console.log('algo paso');
});



socket.on('processed_webrtc', (processed_data) => {
  console.log('llego del server');
  img.src = "data:image/jpg;base64," + processed_data;   
});
