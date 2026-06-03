import WebSocket from 'ws';
const ws1 = new WebSocket('ws://127.0.0.1:8000/ws-test');
ws1.on('open', () => console.log('/ws-test OPEN'));
ws1.on('message', data => console.log('/ws-test MSG', data.toString()));
ws1.on('error', err => console.log('/ws-test ERROR', err.message));
ws1.on('close', code => console.log('/ws-test CLOSE', code));

const ws2 = new WebSocket('ws://127.0.0.1:8000/ws');
ws2.on('open', () => console.log('/ws OPEN'));
ws2.on('message', data => console.log('/ws MSG', data.toString()));
ws2.on('error', err => console.log('/ws ERROR', err.message));
ws2.on('close', code => console.log('/ws CLOSE', code));
