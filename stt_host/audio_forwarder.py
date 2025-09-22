import asyncio
import websockets
import sounddevice as sd
import argparse

SAMPLE_RATE = 16000
FRAME_MS    = 20
FRAME_SMP   = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples

async def stream(uri: str, device_index: int | None):
    async with websockets.connect(uri, ping_interval=20, max_size=None) as ws:
        print("[host] connect:", uri)
        await ws.send("wake_on")

        q = asyncio.Queue()

        def cb(indata, frames, time_info, status):
            if status:
                print("[audio]", status)
            # int16 bytes
            q.put_nowait(indata.copy().tobytes())

        print(f"[host] capturing mic device={device_index} @16kHz mono int16")
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            blocksize=FRAME_SMP, device=device_index, callback=cb):
            while True:
                # 서버 이벤트 수신과 오디오 전송 동시 처리
                recv_task = asyncio.create_task(ws.recv())
                send_task = asyncio.create_task(q.get())
                done, pending = await asyncio.wait(
                    {recv_task, send_task}, return_when=asyncio.FIRST_COMPLETED, timeout=1.0
                )

                if recv_task in done:
                    msg = recv_task.result()
                    if isinstance(msg, bytes):
                        pass
                    else:
                        if msg == "ack:wake_on":
                            print("[server] wake mode on")
                        elif msg == "ack:stt_on":
                            print("[server] stt mode on (continuous)")
                        elif msg == "ack:stt_once":
                            print("[server] stt mode on (single-shot)")
                        elif msg == "event:wake_detected":
                            print("[server] wake word detected → STT once")
                            # 서버가 자동으로 stt_once로 전환하도록 되어 있지만
                            # 확실히 하려면 아래 줄 유지(중복 harmless)
                            await ws.send("stt_once")
                        elif msg == "event:wake_resumed":
                            print("[server] back to wake mode")
                        elif msg.startswith("result:"):
                            text = msg.split("result:",1)[1]
                            print("[server] STT:", text)
                            # TODO: 메인앱 연동(콜백/큐/소켓 등) 넣으려면 여기에서
                        elif msg.startswith("err:"):
                            print("[server]", msg)
                        else:
                            print("[server]", msg)
                else:
                    recv_task.cancel()

                if send_task in done:
                    pcm = send_task.result()
                    await ws.send(pcm)
                else:
                    send_task.cancel()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="ws://127.0.0.1:8000/stream")
    ap.add_argument("--mic", type=int, default=None, help="Windows sounddevice 인덱스(None=기본)")
    args = ap.parse_args()
    asyncio.run(stream(args.ws, args.mic))
