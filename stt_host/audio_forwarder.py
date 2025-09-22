import asyncio
import argparse
import websockets
import sounddevice as sd

# ↓ 추가: 메인 API 콜백 전송용
import httpx

SAMPLE_RATE = 16000
FRAME_MS    = 20
FRAME_SMP   = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples

async def stream(
    uri: str,
    device_index: int | None,
    callback_url: str | None,
    session_id: str | None,
    api_key: str | None,
):
    async with websockets.connect(uri, ping_interval=20, max_size=None) as ws:
        print("[host] connect:", uri)
        await ws.send("wake_on")

        q = asyncio.Queue()

        # 콜백 HTTP 클라이언트 준비
        http_client = None
        if callback_url:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            http_client = httpx.AsyncClient(timeout=15.0, headers=headers)

        def cb(indata, frames, time_info, status):
            if status:
                print("[audio]", status)
            # int16 bytes
            q.put_nowait(indata.copy().tobytes())

        print(f"[host] capturing mic device={device_index} @16kHz mono int16")
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SMP,
                device=device_index,
                callback=cb,
            ):
                while True:
                    # 서버 이벤트 수신과 오디오 전송 동시 처리
                    recv_task = asyncio.create_task(ws.recv())
                    send_task = asyncio.create_task(q.get())
                    done, pending = await asyncio.wait(
                        {recv_task, send_task},
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=1.0
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
                                # 서버가 자동 전환하지만 보수적으로 한 번 더 보냄
                                await ws.send("stt_once")
                            elif msg == "event:wake_resumed":
                                print("[server] back to wake mode")
                            elif msg.startswith("result:"):
                                text = msg.split("result:", 1)[1]
                                print("[server] STT:", text)

                                # ★ 메인 API 콜백
                                if callback_url and http_client:
                                    try:
                                        payload = {"text": text}
                                        if session_id:
                                            payload["session_id"] = session_id
                                        r = await http_client.post(callback_url, json=payload)
                                        # 응답이 길 수 있어서 앞부분만 출력
                                        print("[main_api]", r.status_code, r.text[:200])
                                    except Exception as e:
                                        print("[main_api][ERR]", e)
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
        finally:
            if http_client:
                await http_client.aclose()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="ws://127.0.0.1:8000/stream",
                    help="STT 컨테이너 WebSocket URL")
    ap.add_argument("--mic", type=int, default=None,
                    help="Windows sounddevice 인덱스(None=기본)")
    ap.add_argument("--post", default=None,
                    help="메인 API 콜백 URL (예: http://127.0.0.1:9000/stt/hook). 비우면 콜백 전송 안 함")
    ap.add_argument("--session", default=None,
                    help="세션 ID (대화/사용자 식별용)")
    ap.add_argument("--api-key", default=None,
                    help="메인 API 인증이 필요하면 Bearer 토큰 전달")
    args = ap.parse_args()

    asyncio.run(stream(args.ws, args.mic, args.post, args.session, args.api_key))
