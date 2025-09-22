\
# Simple demo client: upload a dummy file to MinIO via main, then trigger detect.
import os, io, uuid, httpx
from dotenv import load_dotenv
load_dotenv()

MAIN="http://localhost:8000"
LG_TOKEN=os.getenv("LG_API_TOKEN","LG_SECRET")

def run():
    # 1) Upload a dummy "source" file to MinIO
    files = {"file": ("source.jpg", b"\xFF\xD8\xFF\xDB" + b"\x00"*1000 + b"\xFF\xD9", "image/jpeg")}
    r = httpx.post(f"{MAIN}/upload", files=files, timeout=30.0)
    r.raise_for_status()
    url = r.json()["url"]
    print("uploaded source url:", url)

    # 2) Trigger detect on langgraph
    r = httpx.post(f"{MAIN}/trigger/detect", json={"source_url": url}, timeout=30.0)
    r.raise_for_status()
    print("detect ack:", r.json())

if __name__ == "__main__":
    run()