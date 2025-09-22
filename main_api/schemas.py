from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class STTHookIn(BaseModel):
    text: str = Field(..., description="STT가 반환한 텍스트")
    session_id: Optional[str] = Field(None, description="대화/사용자 세션 식별자")
    device_index: Optional[int] = None
    timestamp: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None

class NlpQueryIn(BaseModel):
    text: str
    session_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class NlpResult(BaseModel):
    ok: bool
    text: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
