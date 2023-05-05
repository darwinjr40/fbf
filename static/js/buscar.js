

const socket = io();
const startButton = document.getElementById('start-button');
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");



socket.on('connect', function () {
    console.log('conectados');
    socket.on('event', (res) => {
        console.log(res);
    });    
});

socket.on('processed_buscar_faces',  (res) =>{
    console.log('respuesta de camara server:', res);    
    let img = document.getElementById("imagen");
    img.src = "data:image/jpg;base64," + res;   
});

// socket.on('processed_stream', function(processed_data) {
//   var video = document.getElementById('processed_video');
//   video.src = 'data:image/jpeg;base64,' + processed_data;
// });


startButton.addEventListener('click', () => {
    
  navigator.mediaDevices.getUserMedia({video: true})
  .then(stream => {

    video.srcObject = stream;
    
    video.addEventListener("loadedmetadata", () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    });
    
    setInterval(() => {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      let base64ImageData = canvas.toDataURL("image/png");
      socket.emit('buscarFaces', base64ImageData);
      // socket.emit('buscarFaces', stream);
    // }, 1000/30); // envía 30 frames por segundo
    }, 650); // envía 1 frames por segundo
  })
  .catch(error => {
    console.log("Error accessing camera: " + error.message);
  });

});