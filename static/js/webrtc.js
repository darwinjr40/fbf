const socket = io();
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startButton = document.getElementById('start-button');
const restartButton = document.getElementById('restart-button');
const ctx = canvas.getContext("2d");
const img0 = document.getElementById("imagen0");
const img1 = document.getElementById("imagen1");
const imgs = {
   0: document.getElementById("imagen0"),
   1: document.getElementById("imagen1"),
   2: document.getElementById("imagen2"),
   3: document.getElementById("imagen3"),
  // y así sucesivamente
};
startButton.addEventListener('click', () => {
  socket.emit('start', 'sad');
//     navigator.mediaDevices.getUserMedia({video: true})
//         .then(stream => {
//           video.srcObject = stream;
//           // video.play();          
//           setInterval(() => {
//             ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
//             const base64ImageData = canvas.toDataURL('image/png', 0.5).split(',')[1];
          //  socket.emit('webrtc', base64ImageData);
//             // socket.emit('event', base64ImageData);
// //            console.log(base64ImageData);
//           }, 1000);
//         })
//         .catch(error => {
//             console.log('Error accessing camera:', error.message);
//         });
});
restartButton.addEventListener('click', () => {
  socket.emit('restart', 'sad');
});

video.addEventListener("loadedmetadata", () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  console.log('algo paso');
});



socket.on('processed_webrtc', (processed_data) => {
  console.log('llego del server');
  // console.log('nise');
  // console.log(processed_data['img']);
  // console.log(typeof  processed_data['id']);
  // img1.src = "data:image/jpg;base64," + processed_data['img'];
  imgs[processed_data['id']].src = "data:image/jpg;base64," + processed_data['img'];    
});

// socket.on('processed_webrtc0', (processed_data) => {
//   console.log('llego del server');
//   // console.log('nise');
//   // console.log(processed_data['img']);
//   // console.log(typeof  processed_data['id']);
//   // img1.src = "data:image/jpg;base64," + processed_data['img'];
//   imgs[processed_data['id']].src = "data:image/jpg;base64," + processed_data['img'];    
// });
// socket.on('processed_webrtc1', (processed_data) => {
//   console.log('llego del server');
//   // console.log('nise');
//   // console.log(processed_data['img']);
//   // console.log(typeof  processed_data['id']);
//   // img1.src = "data:image/jpg;base64," + processed_data['img'];
//   imgs[processed_data['id']].src = "data:image/jpg;base64," + processed_data['img'];    
// });

socket.on('connect', function () {
  console.log('conectados');
  // socket.on('event', (res) => {
  //     console.log(res);
  // });    
});

socket.on('event', (processed_data) => {
  console.log('llego del server');
  // console.log(processed_data['img']);
//  img.src = "data:image/jpg;base64," + processed_data['img'];
});
