# **1) 의존성 설치 (Windows PC, VSCode PowerShell 터미널)**

```
# 1) 가상환경
python -m venv venv
.\venv\Scripts\activate

# 2) 필수 패키지
pip install --upgrade pip
pip install opencv-python mediapipe numpy pillow

# 3) PyTorch
pip install --no-cache-dir torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 `
  --index-url https://download.pytorch.org/whl/cu121

pip install --no-cache-dir xformers==0.0.23.post1

# 4) 공통 Python 패키지
pip install --no-cache-dir opencv-python mediapipe numpy pillow

# 4) (선택) ffmpeg CLI 테스트용
#  - OpenCV는 자체 FFmpeg 내장이라 보통 불필요하지만, 테스트용으로 깔아두면 유용
# choco install ffmpeg -y   # Chocolatey 사용 시

```

# **2) Depth-Anything 깃 클론**

VDA(Video-Depth-Anything) 저장소

```
# 원하는 디렉토리에서
git clone https://github.com/DepthAnything/Video-Depth-Anything.git

cd .\Video-Depth-Anything
pip install -r requirements.txt
# (여기서 run_streaming.py가 있어야 함)

```

# **3) 폴더 구조 예시**

```
project_root/
├─ venv/                            # 가상환경
├─ Video-Depth-Anything/            # VDA 저장소 (run_streaming.py 위치)
├─ extrinsic_eye_to_world.json      # 외부파라미터 (eye→worldCam)
├─ gaze_depth_fusion_rtsp.py        # ★ 아래 제공 코드
└─ calib/
   ├─ intrinsics/
   │  ├─ eye/                              # eye 카메라 캡처 이미지
   │  ├─ world/                            # world 카메라 캡처 이미지
   │  ├─ intrinsic_eye.json                # ← solve 결과 (생성)
   │  └─ intrinsic_world.json              # ← solve 결과 (생성)
   └─ extrinsic/
      ├─ pairs/                            # eye/world 동시 캡처 이미지 페어
      └─ solve_log.txt                     # (선택) 로그
└─ (실행 시 생성)
   ├─ _segments/
   └─ _vda_out/

```

# **4. 실행 방법 (Windows PC)**

```
.\venv\Scripts\activate
python gaze_depth_fusion_rtsp.py ^
  --eye_url   rtsp://192.168.162.44:8554/cam ^
  --world_url rtsp://192.168.162.68:8554/cam ^
  --vda_dir   .\Video-Depth-Anything ^
  --encoder   vits --metric --input_size 518 --max_res 1280 ^
  --fx_w <fx> --fy_w <fy> --cx_w <cx> --cy_w <cy> ^
  --extrinsic_json .\extrinsic_eye_to_world.json ^
  --show_fps

```

반드시 PC에서 실행.
Pi는 이미 MediaMTX가 켜져 있고 RTSP를 송출 중이어야 함.
내장 기본 URL은 cam1/cam2 IP로 잡아놨지만, 바뀌면 인자로 넘겨줘.
