// var socket = io.connect(window.location.protocol + '//' + document.domain + ':' + location.port);





// var canvas = document.getElementById('canvas');
// var context = canvas.getContext('2d');
// const video = document.querySelector("#videoElement");

// video.width = 400;
// video.height = 300;

// if (navigator.mediaDevices.getUserMedia) {
//     navigator.mediaDevices.getUserMedia({
//         video: true
//     })
//         .then(function (stream) {
//             video.srcObject = stream;
//             video.play();
//         })
//         .catch(function (err0r) {

//         });
// }


// const FPS = 10;
// setInterval(() => {
//     width = video.width;
//     height = video.height;
//     context.drawImage(video, 0, 0, width, height);
//     var data = canvas.toDataURL('image/jpeg', 0.5);
//     context.clearRect(0, 0, width, height);
//     socket.emit('image', data);
// }, 1000 / FPS);

// socket.on('processed_image', function (image) {
//     photo.setAttribute('src', image);

// });


// var socket = io();



var socket = io();

socket.on('connect', function () {
    console.log('conectados');
    socket.on('event', (res) => {
        console.log(res);
    });    
});

socket.on('processed_stream',  (res) =>{
    console.log('respuesta de camara server:');
    // res = atob(res);
    // Obtener la etiqueta de imagen
    let img = document.getElementById("imagen");

// Asignar el valor del atributo src
    img.src = "data:image/jpg;base64," + res;
    // console.log(decodedString);

});

// socket.on('processed_stream', function(processed_data) {
//   var video = document.getElementById('processed_video');
//   video.src = 'data:image/jpeg;base64,' + processed_data;
// });

navigator.mediaDevices.getUserMedia({video: true})
  .then(stream => {
    var video = document.getElementById("video");
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    
    video.srcObject = stream;
    video.addEventListener("loadedmetadata", () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    });
    
    setInterval(() => {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      var imageData = canvas.toDataURL("image/png");
      socket.emit('stream', imageData);
    // }, 1000/2); // envía 30 frames por segundo
    }, 750); // envía 1 frames por segundo
  })
  .catch(error => {
    console.log("Error accessing camera: " + error.message);
  });
// navigator.mediaDevices.getUserMedia({video: true})
//   .then(stream => {
//     var video = document.getElementById("video");
//     video.srcObject = stream;
//     // console.log(stream)
//     socket.emit('stream', stream);
//     // var streamId = socket.id;
//   })
//   .catch(error => {
//     console.log("Error accessing camera: " + error.message);
//   });