import httpx
from typing import Any, Dict, Optional
from ..config import settings

async def call_langgraph(
    client: httpx.AsyncClient,
    text: str,
    session_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    LangGraph 서버에 POST 요청.
    - 서버마다 바디 포맷이 다를 수 있어 payload는 최대한 일반적으로 구성.
    - 필요 시 이 함수만 수정하면 전체 연동은 그대로 유지.
    """
    payload = {
        "input": text,
        "session_id": session_id,
        "metadata": meta or {},
    }

    headers = {}
    if settings.langgraph_api_key:
        headers["Authorization"] = f"Bearer {settings.langgraph_api_key}"

    r = await client.post(settings.langgraph_url, json=payload, headers=headers)
    r.raise_for_status()
    return r.json()
