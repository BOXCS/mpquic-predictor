import asyncio
import ssl
from aioquic.asyncio import serve, connect
from aioquic.quic.configuration import QuicConfiguration
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.events import StreamDataReceived

class ServerProtocol(QuicConnectionProtocol):
    def quic_event_received(self, event):
        if isinstance(event, StreamDataReceived):
            print(f"Server received: {event.data.decode()}")

class ClientProtocol(QuicConnectionProtocol):
    pass

async def main():
    conf = QuicConfiguration(is_client=False)
    conf.load_cert_chain("cert.pem", "key.pem")
    server = await serve("127.0.0.1", 4433, configuration=conf, create_protocol=ServerProtocol)
    
    client_conf = QuicConfiguration(is_client=True, verify_mode=ssl.CERT_NONE)
    async with connect("127.0.0.1", 4433, configuration=client_conf, create_protocol=ClientProtocol) as client:
        stream_id = client._quic.get_next_available_stream_id()
        client._quic.send_stream_data(stream_id, b"Hello QUIC!", end_stream=True)
        client.transmit()
        await asyncio.sleep(0.5)
    
    server.close()

asyncio.run(main())
