\
import os, io, uuid
from typing import Optional, List, Dict
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from minio import Minio
from datetime import timedelta
import httpx

from dotenv import load_dotenv
load_dotenv()

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "admin12345")
MINIO_BUCKET     = os.getenv("MINIO_BUCKET", "roi")
LANGGRAPH_URL    = os.getenv("LANGGRAPH_URL", "http://localhost:9100")
MAIN_API_TOKEN   = os.getenv("MAIN_API_TOKEN", "MAIN_SECRET")
LG_API_TOKEN     = os.getenv("LG_API_TOKEN", "LG_SECRET")
CALLBACK_BASE    = os.getenv("CALLBACK_BASE", "http://localhost:8000/callbacks")

mc = Minio(MINIO_ENDPOINT.replace("http://","").replace("https://",""),
           access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY,
           secure=MINIO_ENDPOINT.startswith("https://"))

def ensure_bucket():
    found = mc.bucket_exists(MINIO_BUCKET)
    if not found: mc.make_bucket(MINIO_BUCKET)

def put_and_url(obj_name: str, data: bytes, content_type: str="application/octet-stream") -> str:
    ensure_bucket()
    mc.put_object(MINIO_BUCKET, obj_name, io.BytesIO(data), len(data), content_type=content_type)
    from datetime import timedelta
    return mc.get_presigned_url("GET", MINIO_BUCKET, obj_name, expires=timedelta(minutes=30))

app = FastAPI(title="Main (local) API & Callbacks")

# ---------- PUBLIC API (for operator/demo) ----------

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    obj = f"uploads/{uuid.uuid4()}_{file.filename}"
    url = put_and_url(obj, data, file.content_type or "application/octet-stream")
    return {"bucket": MINIO_BUCKET, "object": obj, "url": url}

class TriggerReq(BaseModel):
    source_url: str
    cid: Optional[str] = None

@app.post("/trigger/detect")
async def trigger_detect(req: TriggerReq):
    # Kick off detect on LangGraph with callback_base
    cid = req.cid or f"C-{uuid.uuid4()}"
    base = f"{CALLBACK_BASE}/{cid}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{LANGGRAPH_URL}/detect",
                                 json={"cid": cid, "source_url": req.source_url,
                                       "params": {"conf": 0.25}, "callback_base": base},
                                 headers={"Authorization": f"Bearer {LG_API_TOKEN}"})
        resp.raise_for_status()
        return resp.json()

# ---------- CALLBACKS (LangGraph → Main) ----------

def check_auth(auth: Optional[str]):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(401, "No token")
    token = auth.split(" ", 1)[1]
    if token != MAIN_API_TOKEN:
        raise HTTPException(403, "Bad token")

class NeedROIReq(BaseModel):
    req_id: str
    hint: Optional[dict] = None

@app.post("/callbacks/{cid}/need_roi")
async def need_roi(cid: str, body: NeedROIReq, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    # In a real system, capture/generate latest ROI here.
    # Demo: upload a small placeholder bytes.
    placeholder = b"\xFF\xD8\xFF\xDB" + b"\x00"*500 + b"\xFF\xD9"  # not a real JPEG but a stub
    obj = f"roi/{cid}/roi_{uuid.uuid4()}.jpg"
    url = put_and_url(obj, placeholder, "image/jpeg")
    return {"in_reply_to": body.req_id, "roi_url": url}

class ConfirmDetectionReq(BaseModel):
    req_id: str
    detection_id: str
    question: str
    options: Optional[list] = None
    crop: dict
    deadline: Optional[int] = None

class ConfirmDetectionResp(BaseModel):
    in_reply_to: str
    detection_id: str
    label: str
    confidence: Optional[float] = None
    note: Optional[str] = None

@app.post("/callbacks/{cid}/confirm_detection", response_model=ConfirmDetectionResp)
async def confirm_detection(cid: str, body: ConfirmDetectionReq, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    # TODO: show UI & wait user choice. For demo, auto "cup".
    return ConfirmDetectionResp(in_reply_to=body.req_id, detection_id=body.detection_id, label="cup", confidence=0.98)

class FinalResult(BaseModel):
    in_reply_to: str
    summary: Optional[str] = None
    detections: Optional[list] = None
    artifacts: Optional[dict] = None
    error: Optional[str] = None

@app.post("/callbacks/{cid}/final")
async def final(cid: str, body: FinalResult, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    print(f"[FINAL][{cid}] {body.model_dump()}")
    return {"ok": True}