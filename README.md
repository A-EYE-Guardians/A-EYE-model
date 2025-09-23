# **0) 가상환경 활성화**

C:/Gukbi/Direct_RP_CV/venv/Scripts/Activate.ps1

- pip install -r .\requirements.txt
- pip install --no-deps ultralytics==8.3.26
- pip install --no-deps ultralytics-thop==2.0.17

# **1) intrinsics 로드**

$W = Get-Content .\data\intrinsics_world.json | ConvertFrom-Json
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:KMP_AFFINITY="disabled"

# **2) 실행**

### 2-1. YOLO 도커 (이미 켜둔 상태라면 생략)

### 2-2. STT +LangGraph 도커 켬 (이미 켜둔 상태라면 생략)

### 2-3. (선택)YOLO forwarder 실행 (카메라 0번, 수동 'r'키로 캡처)

```
python .\yolo_host\yolo_forwarder.py `
  --cam 0 `
  --session alpha `
  --lg http://127.0.0.1:8010 `
  --yolo http://127.0.0.1:8090/detect `
  --show
```

> 자동 주기로 돌리고 싶으면(예: 2초마다 추론):

```
python .\yolo_host\yolo_forwarder.py `
  --cam 0 `
  --session alpha `
  --lg http://127.0.0.1:8010 `
  --yolo http://127.0.0.1:8090/detect `
  --interval 2.0
```

### 2-4. (필수)STT, LangGraph audio_forwarder 실행

```
python audio_forwarder.py `
  --ws ws://127.0.0.1:8000/stream `
  --mic 4 `
  --lg http://127.0.0.1:8010/invoke `
  --session alpha `
  --lat 37.5665 `
  --lon 126.9780
```

### 2-5. (필수)main 실행

```
python .\ver2_gaze_yolo_fusion.py `
  --eye_cam 2 --world_cam 1 `
  --av_backend_world dshow --pixel_format_world mjpeg `
  --width_world 512 --height_world 384 --fps_world 30 `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric --input_size 256 --max_res 640 `
  --target_fps 18 --device auto `
  --force_center `
  --fx_w $($W.fx) --fy_w $($W.fy) --cx_w $($W.cx) --cy_w $($W.cy) `
  --Kref_w 512 --Kref_h 384 `
  --extrinsic_json .\data\extrinsic_eye_to_world.json `
  --enable_yolo `
  --yolo_server_url "http://127.0.0.1:8090" `
  --yolo_imgsz 640 --yolo_conf 0.25 --yolo_iou 0.50
```

> yolo-stt-langgraph 합친 버전

```
python .\ver3_gaze_yolo_fusion.py `
  --eye_cam 2 --world_cam 1 `
  --av_backend_world dshow --pixel_format_world mjpeg `
  --width_world 512 --height_world 384 --fps_world 30 `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric --input_size 256 --max_res 640 `
  --target_fps 18 --device auto `
  --force_center `
  --fx_w $($W.fx) --fy_w $($W.fy) --cx_w $($W.cx) --cy_w $($W.cy) `
  --Kref_w 512 --Kref_h 384 `
  --extrinsic_json .\data\extrinsic_eye_to_world.json `
  --enable_yolo `
  --yolo_server_url "http://127.0.0.1:8090" `
  --yolo_imgsz 640 --yolo_conf 0.25 --yolo_iou 0.50 `
  --lg "http://127.0.0.1:8010" `
  --session alpha `
  --gaze_push_hz 2 `
  --show_fps
```

```
python .\ver3_gaze_yolo_fusion.py `
  --eye_cam 2 --world_cam 1 `
  --av_backend_world dshow --pixel_format_world mjpeg `
  --width_world 512 --height_world 384 --fps_world 30 `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric --input_size 256 --max_res 640 `
  --target_fps 18 --device auto `
  --force_center `
  --fx_w $($W.fx) --fy_w $($W.fy) --cx_w $($W.cx) --cy_w $($W.cy) `
  --Kref_w 512 --Kref_h 384 `
  --extrinsic_json .\data\extrinsic_eye_to_world.json `
  --enable_yolo `
  --yolo_server_url "http://127.0.0.1:8090" `
  --yolo_imgsz 640 --yolo_conf 0.25 --yolo_iou 0.50 `
  --lg_url "http://127.0.0.1:8010" `
  --lg_session alpha `
  --show_fps

```

# **기타**

질문 흐름: “방금 찍은 사진 뭐야 / 최근 프레임 설명해줘 / 앞에 뭐 보이니?” → LangGraph가 image_recent/yolo_scene를 자동 호출해서 말로 요약.

“오늘 날씨 어때?” → weather_now(lat, lon) 툴 호출(좌표는 STT → /invoke 호출 시 함께 전달).

R 키: YOLO 1회 실행 → /image/push → /perception/yolo/push
매 ~0.5초(=2Hz): /perception/gaze/push 자동 전송

LangGraph 쪽에서:

- GET /image/recent?session_id=alpha&limit=1
- GET /perception/yolo/recent?session_id=alpha&last=1
- GET /perception/gaze/recent?session_id=alpha&last=1

# 3. STT 호스트 포워더 실행

- cd C:\Gukbi\Direct_RP_CV\stt_host
- pip install websockets sounddevice (최초 한번)

- 콜백 없이 단순 테스트:
  - python audio_forwarder.py --ws ws://127.0.0.1:8000/stream --mic 4
- 메인 API로 콜백 전송

```
python .\audio_forwarder.py `
  --ws ws://127.0.0.1:8000/stream `
  --mic 4 `
  --post http://127.0.0.1:9000/stt/hook `
  --session user-001

```

- API 키가 필요한 경우:

```
python .\audio_forwarder.py `
  --ws ws://127.0.0.1:8000/stream `
  --mic 4 `
  --post http://127.0.0.1:9000/stt/hook `
  --session user-001 `
  --api-key YOUR_TOKEN_HERE

```

# 4. Langgraph 실행

- venv 활성화 후
- uvicorn main_api.app:app --host 0.0.0.0 --port 9000 --reload

```폴더구조
Direct_RP_CV/
├─ .env
├─ main_api/
│  ├─ app.py
│  ├─ config.py
│  ├─ schemas.py
│  └─ clients/
│     ├─ __init__.py
│     └─ langgraph.py
└─ stt_host/
   └─ audio_forwarder.py   # (여기에 STT 결과를 main_api에 POST하는 코드도 추가)

```

확인용 curl

```
# 헬스
curl http://127.0.0.1:9000/health

# 직접 질의
curl -X POST http://127.0.0.1:9000/nlp/query ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"안녕 LangGraph\"}"

```

```
python audio_forwarder.py --ws ws://127.0.0.1:8000/stream --lg http://127.0.0.1:8010/invoke --session alpha
```

```stt 실행하며 langgraph 호출
- cd stt_host

python audio_forwarder.py `
  --ws ws://127.0.0.1:8000/stream `
  --mic 4 `
  --lg http://127.0.0.1:8010/invoke `
  --session alpha `
  --lat 37.5665 `
  --lon 126.9780
```
