import asyncio
import argparse
import websockets
import sounddevice as sd
import httpx
import json
from typing import Optional

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SMP = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples

async def ask_langgraph(
    lg_client: httpx.AsyncClient,
    lg_url: str,
    text: str,
    session_id: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
    debug: bool = True,
) -> str:
    if not text or not text.strip():
        return ""

    payload = {"text": text.strip()}
    if session_id:
        payload["session_id"] = session_id
    if lat is not None and lon is not None:
        payload["lat"] = float(lat)
        payload["lon"] = float(lon)

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json; charset=utf-8",
    }

    try:
        resp = await lg_client.post(lg_url, json=payload, headers=headers)
    except httpx.RequestError as e:
        print("[langgraph][ERR] request error:", repr(e))
        raise

    if resp.status_code >= 400:
        body = resp.text
        print(f"[langgraph][ERR] HTTP {resp.status_code}: {body[:400]}")
        resp.raise_for_status()

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("[langgraph][ERR] non-JSON response head:", resp.text[:400])
        raise

    if debug:
        print("[langgraph][DBG] resp keys:", list(data.keys()))
        try:
            print("[langgraph][DBG] preview:", json.dumps(data, ensure_ascii=False)[:400])
        except Exception:
            pass

    ans = None
    fo = data.get("final_output")
    if isinstance(fo, dict):
        ans = fo.get("answer")
    if ans is None:
        ans = data.get("answer") or ""

    return ans or ""

async def stream(
    uri: str,
    device_index: Optional[int],
    langgraph_url: Optional[str],
    callback_url: Optional[str],
    session_id: Optional[str],
    api_key: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
):
    async with websockets.connect(uri, ping_interval=20, max_size=None) as ws:
        print("[host] connect:", uri)
        await ws.send("wake_on")

        q: asyncio.Queue[bytes] = asyncio.Queue()

        callback_client = None
        langgraph_client = None

        if callback_url:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            callback_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5, read=30, write=30, pool=5),
                headers=headers,
            )

        if langgraph_url:
            langgraph_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5, read=45, write=45, pool=5)
            )

        def cb(indata, frames, time_info, status):
            if status:
                print("[audio]", status)
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
                    recv_task = asyncio.create_task(ws.recv())
                    send_task = asyncio.create_task(q.get())
                    done, pending = await asyncio.wait(
                        {recv_task, send_task},
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=1.0
                    )

                    if recv_task in done:
                        msg = recv_task.result()
                        if not isinstance(msg, str):
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
                                await ws.send("stt_once")
                            elif msg == "event:wake_resumed":
                                print("[server] back to wake mode")
                            elif msg.startswith("result:"):
                                text = (msg.split("result:", 1)[1] or "").strip()
                                print("[server] STT:", text)

                                answer = None
                                if langgraph_url and langgraph_client and text:
                                    try:
                                        answer = await ask_langgraph(
                                            langgraph_client, langgraph_url, text, session_id, lat, lon, debug=True
                                        )
                                        if answer:
                                            print("[langgraph.answer]", (answer[:200] + "…") if len(answer) > 200 else answer)
                                        else:
                                            print("[langgraph.answer] <empty>")
                                    except Exception as e:
                                        print("[langgraph][ERR] exception:", repr(e))

                                if callback_url and callback_client:
                                    try:
                                        cb_payload = {"stt_text": text}
                                        if session_id:
                                            cb_payload["session_id"] = session_id
                                        if answer is not None:
                                            cb_payload["agent_answer"] = answer
                                        headers = {
                                            "Content-Type": "application/json; charset=utf-8",
                                            "Accept": "application/json; charset=utf-8",
                                        }
                                        r = await callback_client.post(callback_url, json=cb_payload, headers=headers)
                                        print("[main_api]", r.status_code, (r.text[:200] + "…") if len(r.text) > 200 else r.text)
                                    except Exception as e:
                                        print("[main_api][ERR]", repr(e))
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
            if callback_client:
                await callback_client.aclose()
            if langgraph_client:
                await langgraph_client.aclose()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="ws://127.0.0.1:8000/stream",
                    help="STT 컨테이너 WebSocket URL")
    ap.add_argument("--mic", type=int, default=None,
                    help="Windows sounddevice 인덱스(None=기본)")
    ap.add_argument("--lg", default="http://127.0.0.1:8010/invoke",
                    help="LangGraph FastAPI /invoke URL")
    ap.add_argument("--post", default=None,
                    help="(옵션) 메인 API 콜백 URL")
    ap.add_argument("--session", default="alpha",
                    help="세션 ID (대화/사용자 식별용)")
    ap.add_argument("--api-key", default=None,
                    help="메인 API 인증용 Bearer 토큰")
    ap.add_argument("--lat", type=float, default=None, help="위도")
    ap.add_argument("--lon", type=float, default=None, help="경도")
    args = ap.parse_args()

    asyncio.run(stream(
        args.ws,
        args.mic,
        args.lg,
        args.post,
        args.session,
        args.api_key,
        args.lat,
        args.lon
    ))
