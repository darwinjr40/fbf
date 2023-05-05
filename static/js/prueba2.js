var socket = io();
// var socket = io('http://192.168.1.21:5000');


sendMsj = () => {
    socket.emit('event', 'hola');
};

socket.on('connect', function () {
    console.log('conectados');
    socket.on('event', (res) => {
        console.log(res);
    });
    // socket.emit('my event', { data: 'I\'m connected!' });
});


// socket.on('disconnect', function () {
//     console.log("Desconectados !");
// });

