const http = require('http')
, express = require('express')
, io = require('socket.io')
, app = express();

const server = http.createServer(app).listen(80);

app.get('/', (req, res) => {

});
