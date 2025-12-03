<div align="center">

# A-EYE  
**Gaze + Depth + YOLO + LLM 기반 실시간 Multimodal AI Agent**

> 시선 추적과 객체 인식을 결합해 사용자의 ‘관심’을 인지하고 대화하는  
> **LangGraph 기반 실시간 멀티모달 AI 시스템**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-000000?style=flat&logo=github&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)]()


> **빠른 실행 테스트는 4번 "Setup & Run" 항목을 확인해주세요!**
</div>

---

## 1. Overview  

**A-EYE**는 시선 추적(eye-tracking), 깊이맵 추정(depth estimation), 객체 인식(object detection),  
그리고 LLM 기반 자연어 이해를 통합한 **멀티모달 AI 에이전트**입니다.  

> 사용자가 ‘무엇을 바라보는가’를 인식하고, 그 시선과 장면 정보를 LangGraph 기반 LLM에 전달하여  
> 실시간으로 대화형 피드백을 생성합니다.  

**목적**  
- 카메라 시점 이해와 음성 대화 기능을 통합한 실시간 인터랙티브 AI  
- 시각 정보 기반 인간-AI 상호작용 연구 및 서비스화 기반 구축  

**핵심 특징**  
- 단일 시선 카메라 + 월드 카메라 융합  
- Depth-Anything v2 Metric 기반 절대 깊이 추정  
- YOLOE 객체 탐지 + LangGraph-LLM 대화 파이프라인  
- Docker 기반 마이크로서비스 구조로 확장성 확보  

---


## 2. Architecture  
```
┌────────────┐ ┌──────────────┐ ┌────────────┐
│ Eye Cam │ ─▶── │ Gaze Vector │ ─▶── │ │
└────────────┘ │ (Mediapipe) │ │ │
└──────┬───────┘ │ │
▼ │ │
┌────────────┐ ┌──────────────┐ ┌────────────┐
│ World Cam │ ─▶── │ Depth Map │ ─▶── │ YOLOE │ ─▶ JSON
└────────────┘ │ (Depth-Anything v2) │ └────────────┘
│
▼
┌────────────────────────┐
│ LangGraph-LLM Agent │
│ Gaze + Object + Context │
└────────────────────────┘
│
▼
💬 Real-time Response
```
**Backend** : FastAPI + PyTorch + LangGraph  
**Infrastructure** : Docker Compose, GPU (RTX 3050 Ti 4GB)  
**Models** : YOLOE, Depth-Anything v2, Whisper-STT, LangGraph-LLM  

---


## 3. Features  

| 기능 | 설명 | 주요 기술 |
|------|------|------------|
|  시선 추적 | Mediapipe 기반 홍채 중심 벡터 계산 | OpenCV, Mediapipe |
|  깊이맵 추정 | Depth-Anything v2 Metric 모델 사용 | PyTorch, LiheYoung Repo |
|  객체 인식 | YOLOE로 객체 탐지 | Ultralytics YOLO |
|  LangGraph Agent | 시선+객체+텍스트 기반 응답 생성 | FastAPI, LangGraph |
|  마이크로서비스 구조 | YOLO/STT/LLM을 Docker 컨테이너로 분리 | Docker Compose |

---


## 4. Setup & Run

A-EYE는 **멀티 레포지토리 모노레포 스타일**로 구성되어 있습니다. 아래 3개 레포를 각각 클론하고, 각 레포의 `README`를 따라 세팅하세요.

### 1) Repositories

- **Model (메인 서버 + 카메라 모듈 + Wakeword + TTS)**  
  👉 [A-EYE-Guardians/A-EYE-model (hayong-stt)](https://github.com/A-EYE-Guardians/A-EYE-model/tree/hayong-stt)

- **YOLOe (객체 탐지 / Docker 서버, 메인 서버와 HTTP 통신)**  
  👉 [A-EYE-Guardians/A-EYE-YOLOe (hayong)](https://github.com/A-EYE-Guardians/A-EYE-YOLOe/tree/hayong)

- **LangGraph (LLM Agent / Docker 서버, 메인 서버와 HTTP 통신)**  
  👉 [A-EYE-Guardians/A-EYE-LANGGRAPH (hayong)](https://github.com/A-EYE-Guardians/A-EYE-LANGGRAPH/tree/hayong)

> A-EYE 통합 실행 시에는 **Model**이 클라이언트/오케스트레이터 역할을 수행합니다.

---

### 2) Clone

```powershell
# 작업 폴더 생성
mkdir A-EYE && cd A-EYE

# 3개 레포 각각 클론 (브랜치 포함)
git clone --branch hayong-stt https://github.com/A-EYE-Guardians/A-EYE-model.git
git clone --branch hayong     https://github.com/A-EYE-Guardians/A-EYE-YOLOe.git
git clone --branch hayong     https://github.com/A-EYE-Guardians/A-EYE-LANGGRAPH.git
```

### 3) Per-Repo Setup (각 레포 README를 먼저 확인)
**A-EYE-model (Main)**

- 역할: 메인 서버 + 카메라 모듈 + Wakeword + STT + TTS, 그리고 YOLOe / LangGraph와 통신
- 기본 작업:

  - Python env 구성, 카메라/오디오 장치 설정
  - (옵션) 로컬 GPU 사용 시 PyTorch (CUDA) 설치
  - .env 또는 설정 파일에서 외부 서비스 URL/포트 지정

**A-EYE-YOLOe (Detection Server / Docker)**

- 역할: 객체 탐지 전용 Docker 서버, 메인 서버에서 HTTP로 호출
- 기본 작업:

  - Docker 이미지 빌드, 가중치 경로 마운트
  - YOLO_WEIGHTS, YOLO_IMGSZ, YOLO_CONF 등 환경변수 설정
  - 컨테이너 포트 공개 (기본: 8090)

**A-EYE-LANGGRAPH (LLM Agent / Docker)**

- 역할: LangGraph 기반 멀티모달 에이전트 서버, 메인 서버에서 프레임/탐지결과/음성결과를 전송 받아 응답 생성
- 기본 작업:

  - Docker 이미지 빌드
  - 모델 키/엔드포인트(필요 시) 및 내부 플로우 설정
  - 컨테이너 포트 공개 (기본: 8010)
 
### 4) 네트워킹 & 환경변수 (권장)
포트/URL 컨벤션

- YOLOe: http://127.0.0.1:8090
- LangGraph: http://127.0.0.1:8010
- Model(Main): 로컬 실행 프로세스 (필요 시 내부 API 포트 사용)
```
# External services
YOLO_SERVER_URL=http://127.0.0.1:8090
LANGGRAPH_URL=http://127.0.0.1:8010

# Cameras / Audio
WORLD_CAM_INDEX=1
EYE_CAM_INDEX=2
SAMPLE_RATE=16000

# STT / TTS
WHISPER_MODEL=small
TTS_ENGINE=gtts

# Performance
TARGET_FPS=18
MAX_RES=640

```

### 5) 실행 순서 (권장 시나리오)

1. YOLOe 서버 기동 (Docker)

  - 가중치 마운트 및 환경변수 확인
  - 예: docker run -p 8090:8090 ... --name yoloe ...

2. LangGraph 서버 기동 (Docker)

  - LangGraph 플로우/엔드포인트 정상 기동
  - 예: docker run -p 8010:8010 ... --name langgraph ...

3. Model(Main) 로컬 실행

  - 카메라/오디오 장치 확인 후 메인 파이프라인 시작
  - 예:
```
# 1. stt 가동

cd C:\Gukbi\Direct_RP_CV\stt_host

python audio_forwarder.py `
  --ws ws://127.0.0.1:8000/stream `
  --mic 2 `
  --lg http://127.0.0.1:8010/invoke `
  --session alpha `
  --lat 37.5665 `
  --lon 126.9780

# 2. main (ver4_gaze_yolo_fusion.py) 실행

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

### 6) 데이터 플로우 (요약)
```
[Model(Main)]
  ├─ 캡처: EyeCam/WorldCam + 오디오(Wakeword → STT)
  ├─ 요청: YOLOe 서버로 프레임 전송 → 탐지 결과 수신
  ├─ 전송: LangGraph 서버로 프레임/탐지/음성결과 전달
  └─ 응답: LangGraph 결과 수신 → TTS 출력/화면 표시

```

### 7) Sanity Check (필수 점검)

- YOLOe /health 또는 /docs(있다면) 접속 확인
- LangGraph /health 또는 /docs 확인
- Model 실행 시 카메라 인덱스, 마이크 장치, FPS 로그 정상 출력
- Model → YOLOe → LangGraph 호출 시 HTTP 200 응답 확인

### 8) Troubleshooting (퀵 팁)

- 카메라 지연/끊김: 해상도/프레임 낮추고, RTSP/AV 옵션(rtsp_transport=tcp, stimeout) 확인
- GPU 메모리 부족(4GB): 모델 입력 크기 축소(--max_res 640 → 512), 배치=1 고정
- CORS/포트 충돌: Docker 포트 재매핑 또는 방화벽 규칙 점검
- 지연이 큰 경우: YOLOe/Depth 추론 해상도와 빈도 낮추기, LangGraph 메시지 페이로드 최소화

### Model Weights

| Model             | Source                                                                        |
| ----------------- | ----------------------------------------------------------------------------- |
| YOLOE        | [ultralytics/yolo](https://github.com/THU-MIG/yoloe)                                                         |
| Depth-Anything v2 | [LiheYoung/Depth-Anything-V2](https://github.com/LiheYoung/Depth-Anything-V2) |
| Whisper-small     | `openai/whisper-small`                                                        |
| LangGraph LLM     | Local API endpoint (`http://127.0.0.1:8010`)                                  |

---


## 5. How It Works

- 시선 벡터 계산: Mediapipe로 홍채 중심을 검출하고 EyeCam 기준 시선벡터 추정
- 깊이 추정: Depth-Anything v2로 월드카메라 입력의 절대 깊이맵 산출
- 객체 인식: YOLOE가 프레임 내 주요 객체 탐지
- 데이터 융합: 시선-깊이-객체 데이터를 LangGraph Agent에 전달
- LLM 응답: 문맥 기반 자연어 응답 생성 → 실시간 TTS 출력

---


## 6. Results

| 항목           | 결과                         |
| ------------ | -------------------------- |
| FPS          | 평균 20–27 fps (RTX 3050 Ti) |
| YOLO 추론 시간   | 45–60 ms                   |
| Depth 추론 시간  | 80 ms                      |
| 시선–객체 융합 성공률 | 약 92% (테스트 샘플 기준)          |

---


## 7. Future Work

- **LangGraph 내 MOE(Mixture of Experts) 구조 도입**  
  - LangGraph 파이프라인을 전문가 모델(MOE) 기반으로 확장  
  - 이미지 관련 질의는 **VQA 모델**, 그 외 일반 질의는 **GPT-4o-mini**가 담당하도록 분기  
  - 작업 부하를 모델 특성에 맞게 분산해 응답 정확도와 효율을 동시 향상  

- **시선 추정·깊이맵 동시 추론 모델 구성 및 학습**  
  - 현재 구조는 시선 추정과 깊이 추론을 **별도 프로세스**로 수행 → 시간 지연 및 정합성 한계 존재  
  - RT-GENE, Gaze360 기반 **시계열 시선 데이터**와 Video-Depth 기반 **시계열 깊이 데이터**를 통합  
  - 실제 시선 좌표 및 깊이 정보를 label로 사용해 **하나의 딥러닝 모델**에서 동시 추론 가능하도록 학습  

- **On-Device 연결 및 Edge 추론 최적화**  
  - 경량 모델 변환(TensorRT, OpenVINO)을 통한 Edge 환경 실시간 처리  
  - Raspberry Pi·Jetson 등 임베디드 디바이스에서의 실행 성능 최적화  

- **LangGraph Reflection 기반 응답 품질 고도화**  
  - Reflection 노드를 추가하여 LLM의 내부 reasoning과 output 검증 단계 구현  
  - 사용자 피드백·시선 정보·컨텍스트를 기반으로 **대화 품질과 일관성 향상**

---


## 8. Team

| 이름                | 역할                         | 기여 내용                                |
| ----------------- | -------------------------- | ------------------------------------ |
| **양하용**           | Project Lead | 시선–깊이 융합 알고리즘, LangGraph 통합, Docker화 |
| **김동현** | Backend                    | FastAPI 서버 및 데이터 파이프라인               |
| **이찬환** | yolo model                   | YOLOE 프로토타입 테스트 및 연동            |
| **최리준** | langGraph model                   | LangGraph + STT 파이프라인 구성            |

---


## 9. Reference & License

- [Depth-Anything v2 (LiheYoung)](https://github.com/LiheYoung/Depth-Anything-V2)  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- [LangGraph](https://github.com/langchain-ai/langgraph)  

---

© 2025 **A-EYE Guardians Team**  
All Rights Reserved.  
Distributed under the **MIT License**.

  <div align="center">  <i>“A-EYE는 단순한 시선 인식이 아닌, ‘AI가 세상을 바라보는 방식’을 설계한 프로젝트입니다.”</i> </div> 
