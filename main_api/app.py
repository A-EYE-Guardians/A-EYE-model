import anyio
import httpx
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from .schemas import STTHookIn, NlpQueryIn, NlpResult
from .config import settings
from .clients.langgraph import call_langgraph

app = FastAPI(title="Direct_RP_CV Main API")

# CORS (필요 시 도메인 제한)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: 운영 시 특정 Origin만
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# httpx 클라이언트 (재사용)
_client: httpx.AsyncClient | None = None

@app.on_event("startup")
async def _startup():
    global _client
    timeout = httpx.Timeout(settings.http_timeout_s)
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
    _client = httpx.AsyncClient(timeout=timeout, limits=limits)

@app.on_event("shutdown")
async def _shutdown():
    global _client
    if _client:
        await _client.aclose()
        _client = None

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/nlp/query", response_model=NlpResult)
async def nlp_query(body: NlpQueryIn):
    assert _client is not None
    try:
        data = await call_langgraph(_client, body.text, body.session_id, body.meta)
        # LangGraph 응답 구조가 다양할 수 있으니 text 추출은 유연하게 처리
        text = (
            data.get("text")
            or data.get("output")
            or data.get("result")
            or data.get("message")
        )
        return NlpResult(ok=True, text=text, raw=data)
    except httpx.HTTPStatusError as e:
        return NlpResult(ok=False, error=f"LangGraph {e.response.status_code}: {e.response.text}")
    except Exception as e:
        return NlpResult(ok=False, error=str(e))

@app.post("/stt/hook", response_model=NlpResult, status_code=status.HTTP_200_OK)
async def stt_hook(body: STTHookIn):
    """
    STT Host(호스트의 audio_forwarder)가 STT 결과를 넘겨주는 엔드포인트.
    → 즉시 LangGraph 호출 → LangGraph 응답을 리턴.
    """
    if not body.text or not body.text.strip():
        return NlpResult(ok=False, error="empty STT text")

    assert _client is not None

    # 간단한 재시도(네트워크 일시 오류 대비)
    last_err = None
    for attempt in range(max(1, settings.http_retries)):
        try:
            data = await call_langgraph(_client, body.text, body.session_id, body.meta)
            text = (
                data.get("text")
                or data.get("output")
                or data.get("result")
                or data.get("message")
            )
            return NlpResult(ok=True, text=text, raw=data)
        except Exception as e:
            last_err = e
            # 짧은 백오프
            await anyio.sleep(0.3 * (attempt + 1))

    return NlpResult(ok=False, error=str(last_err) if last_err else "unknown error")
