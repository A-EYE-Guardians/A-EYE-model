<div align="center">

# 🧠 A-EYE  
**Gaze + Depth + YOLO + LLM 기반 실시간 Multimodal AI Agent**

> 시선 추적과 객체 인식을 결합해 사용자의 ‘관심’을 인지하고 대화하는  
> **LangGraph 기반 실시간 멀티모달 AI 시스템**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-000000?style=flat&logo=github&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)]()

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
**Models** : YOLOv8-seg, Depth-Anything v2, Whisper-STT, LangGraph-LLM  

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

## 4. Installation  

### Requirements  
```
Python 3.11.9
PyTorch 2.7.0 + CUDA 11.8
Windows 10 / WSL2 / Ubuntu 22.04
GPU: RTX 3050 Ti (4GB)
```

### Setup  
```
git clone https://github.com/hayongyang/A-EYE.git
cd A-EYE
pip install -r requirements.txt
```

### Model Weights

| Model             | Source                                                                        |
| ----------------- | ----------------------------------------------------------------------------- |
| YOLOE        | `[ultralytics/yolo](https://github.com/THU-MIG/yoloe)`                                                          |
| Depth-Anything v2 | [LiheYoung/Depth-Anything-V2](https://github.com/LiheYoung/Depth-Anything-V2) |
| Whisper-small     | `openai/whisper-small`                                                        |
| LangGraph LLM     | Local API endpoint (`http://127.0.0.1:8010`)                                  |



### Run Example
```
python gaze_yolo_fusion.py \
  --world_cam 1 --eye_cam 2 \
  --vda_dir "./Video-Depth-Anything" \
  --encoder vits --metric --input_size 256 \
  --yolo_server_url "http://127.0.0.1:8090" \
  --lg_url "http://127.0.0.1:8010"

```

## 5. How It Works

- 시선 벡터 계산: Mediapipe로 홍채 중심을 검출하고 EyeCam 기준 시선벡터 추정

- 깊이 추정: Depth-Anything v2로 월드카메라 입력의 절대 깊이맵 산출

- 객체 인식: YOLOv8-seg가 프레임 내 주요 객체 탐지 및 마스크 추출

- 데이터 융합: 시선-깊이-객체 데이터를 LangGraph Agent에 전달

- LLM 응답: 문맥 기반 자연어 응답 생성 → 실시간 TTS 출력

## 6. Results

| 항목           | 결과                         |
| ------------ | -------------------------- |
| FPS          | 평균 17–20 fps (RTX 3050 Ti) |
| YOLO 추론 시간   | 45–60 ms                   |
| Depth 추론 시간  | 80 ms                      |
| 시선–객체 융합 성공률 | 약 92% (테스트 샘플 기준)          |


Demo: 🔗 영상 보기 (예시)

## 7. Future Work

- 시선 추정 딥러닝 모델 직접 학습 (RT-GENE / Gaze360 기반)

- On-Device Edge 추론 최적화 (TensorRT, OpenVINO)

- LangGraph Reflection 기반 대화 지속성 향상

## 8. Team

| 이름                | 역할                         | 기여 내용                                |
| ----------------- | -------------------------- | ------------------------------------ |
| **양하용**           | Project Lead | 시선–깊이 융합 알고리즘, LangGraph 통합, Docker화 |
| **김동현** | Backend                    | FastAPI 서버 및 데이터 파이프라인               |
| **이찬환** | yolo model                   | YOLOE 프로토타입 테스트 및 연동            |
| **최리준** | langGraph model                   | LangGraph + STT 파이프라인 구성            |

## 9. Reference & License

- Depth-Anything v2 (LiheYoung)

- Ultralytics YOLOE

- LangGraph

- License: team A-EYE

  <div align="center"> 💬 <i>“A-EYE는 단순한 시선 인식이 아닌, ‘AI가 세상을 바라보는 방식’을 설계한 프로젝트입니다.”</i> </div> ```
