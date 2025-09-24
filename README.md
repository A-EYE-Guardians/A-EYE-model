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
  --lon 126.9780 `
  --no-tts
```

# ** Final !!최종 실행 방법!!**

### 1. stt 가동

```
cd C:\Gukbi\Direct_RP_CV\stt_host

python audio_forwarder.py `
  --ws ws://127.0.0.1:8000/stream `
  --mic 2 `
  --lg http://127.0.0.1:8010/invoke `
  --session alpha `
  --lat 37.5665 `
  --lon 126.9780
```

### 2. main (ver4_gaze_yolo_fusion.py) 실행

- 또 다른 powershell 열고 루트에서 실행하기

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

### 3. main 별도 버전들

```이거는 나중에 홍채 인식 필요하면 켜기
python .\ver4_gaze_yolo_fusion.py `
  --eye_cam 1 --world_cam 0 `
  --av_backend_eye dshow --pixel_format_eye mjpeg `
  --av_backend_world dshow --pixel_format_world mjpeg `
  --width_eye 320 --height_eye 240 --fps_eye 30 `
  --width_world 512 --height_world 384 --fps_world 30 `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric --input_size 256 --max_res 640 `
  --target_fps 18 --device auto `
  --fx_w $($W.fx) --fy_w $($W.fy) --cx_w $($W.cx) --cy_w $($W.cy) `
  --Kref_w 512 --Kref_h 384 `
  --extrinsic_json .\data\extrinsic_eye_to_world.json `
  --enable_yolo `
  --yolo_server_url "http://127.0.0.1:8090" `
  --yolo_imgsz 640 --yolo_conf 0.25 --yolo_iou 0.50 `
  --lg_url "http://127.0.0.1:8010" `
  --lg_session alpha `
  --show_eye `
  --show_fps
```

# ** 시연 팁**

### 1) 실행 순서 & 명령어

- 1. 오디오(STT) 포워더

```
python audio_forwarder.py `
  --ws ws://127.0.0.1:8000/stream `
  --mic 4 `
  --lg http://127.0.0.1:8010/invoke `
  --session alpha `
  --lat 37.5665 `
  --lon 126.9780 `
  --no-tts

```

- 역할: 음성 → 텍스트(“한 번 묻고 한 번 답하는” 싱글샷) → /invoke로 질문 보냄

- 2. 시선×깊이 메인(Ver4)

```
python .\ver4_gaze_yolo_fusion.py `
  --world_cam 1 --eye_cam 2 `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric --input_size 256 --max_res 640 `
  --target_fps 18 --device auto `
  --force_center `
  --fx_w $W.fx --fy_w $W.fy --cx_w $W.cx --cy_w $W.cy `
  --Kref_w 512 --Kref_h 384 `
  --extrinsic_json .\data\extrinsic_eye_to_world.json `
  --enable_yolo `
  --yolo_stream `
  --yolo_stream_interval 2.0 `
  --yolo_stream_min_conf 0.30 `
  --yolo_server_url "http://127.0.0.1:8090" `
  --yolo_imgsz 640 --yolo_conf 0.25 --yolo_iou 0.50 `
  --lg_url "http://127.0.0.1:8010" --lg_session alpha `
  --show_fps


```

- 역할: 0.5s 간격으로 /perception/gaze/push (정규화 좌표 + 거리)
- 핫키(월드 창): R = YOLO one-shot(현재 프레임 업로드 + bbox 업링크)

> 팁: 세 프로세스 모두 --session alpha로 맞추면 LangGraph가 같은 “대화 컨테이너”로 묶습니다.

2. 데모 대사 & 액션 시나리오

### 시나리오 A — “방금 사진 뭐야?”

액션: 월드 창에서 R 누르기 → YOLO 결과와 이미지가 서버에 올라감
말하기(마이크): “방금 사진 뭐야?”
기대 반응:
graph.py가 yolo_scene(last=1) → 필요시 image_recent(limit=1) → 이미지 전체 요약 발화
예: “핵심: 책상 위에 모니터가 크게 보입니다. 첫째, 화면이 정면에 가깝고… 둘째, 주변에 키보드… 셋째, 좌측에 작은 물체가 있습니다.”
(OPENAI_API_KEY가 있으면 비전 요약, 없으면 YOLO 폴백 요약)

### 시나리오 B — “앞에 뭐 보이니?”

액션: 다시 R (새 프레임 업로드)
말하기: “앞에 뭐 보이니?”
기대 반응: 가장 최근 YOLO/이미지 기반 장면 요약 + 필요시 보조 설명

### 시나리오 C — “나랑 얼마나 떨어졌어?” or “지금 위험한 거 있어?”

액션: Ver4가 자동으로 /perception/gaze/push 중(0.5s)
말하기: “지금 위험한 거 있어?”
기대 반응: gaze_depth_status(last=1)를 읽고 거리/주의 간단 브리핑
예: “핵심: 정면 약 2미터 정도입니다. 첫째, 가까워지면 속도를 줄이세요. 둘째, 좌우 시야도 확인하세요.”

### 시나리오 D — “날씨 어때?”

말하기: “지금 날씨 어때?”
기대 반응: /invoke가 직전에 전달된 좌표(lat,lon)로 weather_now 호출 후 현재 날씨 안내

---

```ver5_gaze_yolo_fusion.py

python .\ver5_gaze_yolo_fusion.py `
  --eye_cam 2 `
  --world_cam 1 `
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
  --show_fps `
  --yolo_stream `                # 시작할 때부터 상시 탐지 ON
  --yolo_stream_overlay `        # 박스 오버레이도 시작 시 ON
  --yolo_stream_push_lg          # 탐지결과를 LG로도 push 시작

한 대 카메라로 상시 전체 YOLO 스트리밍 + ROI 원샷(R키) + 런타임 토글(S/O/P) 모두 지원합니다.

S: 스트리밍 on/off
O: 박스 오버레이 on/off
P: 스트리밍 결과를 LangGraph로 push on/off
R: ROI one-shot (현재 화면 중심 또는 계산된 fixation 기준)
기존 Ver4 기능(가제/깊이, gaze push, 이미지/YOLO 업로드) 그대로 유지됩니다.
```
