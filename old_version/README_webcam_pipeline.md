전체 워크플로우 (요약 → 명령 예시 포함) 0) 프로젝트 준비

- 폴더

```
gaze_depth_fusion/
├─ Video-Depth-Anything/
├─ _segments/                 # 자동 생성
├─ _vda_out/                  # 자동 생성
├─ data/
│  ├─ intrinsics_world.json
│  ├─ intrinsics_eye.json
│  └─ extrinsic_eye_to_world.json
├─ calib/                     # 캘리브 샷 저장
│  └─ intrinsics/
│     ├─ world/*.jpg
│     └─ eye/*.jpg
├─ scripts/
│  ├─ calib_intrinsics.py
│  └─ estimate_extrinsic_dual_chessboard.py
└─ gaze_depth_fusion_webcam.py

```

- 가상환경 & 패키지

```
python -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install opencv-python numpy mediapipe matplotlib

# (GPU) PyTorch: 환경에 맞춰 설치

```

-VDA 준비: Video-Depth-Anything 클론 + run_streaming.py에 depth npz 저장 1줄 패치 권장.

---

1. Intrinsics (체커보드, 각 카메라별)
   (1) 캡처

월드 카메라 샷(웹캠 예시, dshow):

```
python scripts/calib_intrinsics.py grab ^
  --src "video=Iriun Webcam #2" ^
  --av_backend dshow --pixel_format mjpeg ^
  --width 512 --height 384 --fps 30 --flip 0 ^
  --save_dir calib/intrinsics/world ^
  --chess_cols 9 --chess_rows 6

```

아이 카메라 샷:

```
cmd /c ^
python scripts\calib_intrinsics.py grab ^
  --src "video=Iriun Webcam" ^
  --av_backend dshow --pixel_format mjpeg ^
  --width 640 --height 480 --fps 30 --flip 0 ^
  --save_dir calib\intrinsics\eye ^
  --chess_cols 9 --chess_rows 6


```

팁: 다양한 거리/각도로 20~30장. 자동캡처 E, 수동 SPACE.

(2) 해결(내부 파라미터 추정)

```
# 월드
python scripts/calib_intrinsics.py solve ^
  --img_glob "calib/intrinsics/world/*.jpg" ^
  --chess_cols 9 --chess_rows 6 --square_m 0.025 ^
  --out_json data/intrinsics_world.json ^
  --write_preview calib/intrinsics/world/_undist_preview.jpg

# 아이
python scripts/calib_intrinsics.py solve ^
  --img_glob "calib/intrinsics/eye/*.jpg" ^
  --chess_cols 9 --chess_rows 6 --square_m 0.025 ^
  --out_json data/intrinsics_eye.json ^
  --write_preview calib/intrinsics/eye/_undist_preview.jpg

```

결과 JSON에 K + fx/fy/cx/cy + dist 기록됨.

2. Extrinsic (eye→world, 2-보드 지그: A=월드, B=아이)

지그: 두 보드 앞면이 서로 평행·정반대. 앞면-앞면 간격 thickness_m 정확히 측정.

스크립트: estimate_extrinsic_dual_chessboard.py (웹캠/RTSP 모두 지원)

```
python scripts/estimate_extrinsic_dual_chessboard.py ^
  --eye_src "video=Iriun Webcam" ^
  --world_src "video=USB2.0 Camera" ^
  --av_backend_eye dshow --pixel_format_eye mjpeg ^
  --av_backend_world dshow --pixel_format_world mjpeg ^
  --width_eye 640 --height_eye 480 --fps_eye 30 ^
  --width_world 512 --height_world 384 --fps_world 30 ^
  --eye_intrinsic data/intrinsics_eye.json ^
  --world_intrinsic data/intrinsics_world.json ^
  --rows_a 6 --cols_a 9 --square_a 0.025 ^
  --rows_b 6 --cols_b 9 --square_b 0.025 ^
  --assume_back_to_back --thickness_m 0.060 ^
  --max_samples 30 --auto --auto_interval 0.7 ^
  --out_json data/extrinsic_eye_to_world.json

```

키: SPACE=샘플, E=자동 토글, ENTER=계산/저장, BACKSPACE=되돌리기, ESC/q=종료.

출력: data/extrinsic_eye_to_world.json (R 3×3, t 3×1, 미터)

3. 메인 실행 (시선×깊이 융합, 웹캠)

```
# PowerShell
$W = Get-Content .\data\intrinsics_world.json | ConvertFrom-Json

python .\gaze_depth_fusion_webcam.py `
  --eye_cam 2 --world_cam 1 --backend dshow `
  --width_eye 320 --height_eye 240 --fps_eye 30 --fourcc_eye MJPG `
  --width_world 512 --height_world 384 --fps_world 24 --fourcc_world MJPG `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric --input_size 392 --max_res 960 `
  --segment_secs 1 --target_fps 20 --show_fps `
  --fx_w $($W.fx) --fy_w $($W.fy) --cx_w $($W.cx) --cy_w $($W.cy) `
  --extrinsic_json .\data\extrinsic_eye_to_world.json


```

python scripts/calib_intrinsics.py grab ^
--src 1 --backend dshow --width 512 --height 384 --fps 30 --fourcc MJPG ^
--save_dir calib/intrinsics/world ^
--chess_cols 9 --chess_rows 6

python scripts/calib_intrinsics.py grab ^
--src 2 --backend dshow --width 320 --height 240 --fps 30 --fourcc MJPG ^
--save_dir calib/intrinsics/eye ^
--chess_cols 9 --chess_rows 6

```
python scripts/estimate_extrinsic_dual_chessboard.py ^
  --eye_src 2 ^
  --world_src 1 ^
  --av_backend_eye dshow --pixel_format_eye mjpeg ^
  --av_backend_world dshow --pixel_format_world mjpeg ^
  --width_eye 320 --height_eye 240 --fps_eye 30 ^
  --width_world 512 --height_world 384 --fps_world 30 ^
  --eye_intrinsic data/intrinsics_eye.json ^
  --world_intrinsic data/intrinsics_world.json ^
  --rows_a 6 --cols_a 9 --square_a 0.025 ^
  --rows_b 6 --cols_b 9 --square_b 0.025 ^
  --assume_back_to_back --thickness_m 0.060 ^
  --max_samples 30 --auto --auto_interval 0.7 ^
  --out_json data/extrinsic_eye_to_world.json
```

```
python .\gaze_depth_fusion_webcam.py `
  --eye_cam 2 --world_cam 1 --backend dshow `
  --width_eye 320 --height_eye 240 --fps_eye 30 --fourcc_eye MJPG `
  --width_world 448 --height_world 336 --fps_world 20 --fourcc_world MJPG `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric `
  --input_size 256 `
  --max_res 640 `
  --segment_secs 1 `
  --target_fps 12 `
  --show_fps `
  --fx_w $($W.fx) --fy_w $($W.fy) --cx_w $($W.cx) --cy_w $($W.cy) `
  --extrinsic_json .\data\extrinsic_eye_to_world.json
```

```cuda 설정 + 저메모리 파라미터
$W = Get-Content .\data\intrinsics_world.json | ConvertFrom-Json

python .\gaze_depth_fusion_webcam.py `
  --eye_cam 2 --world_cam 1 --backend dshow `
  --width_eye 320 --height_eye 240 --fps_eye 30 --fourcc_eye MJPG `
  --width_world 448 --height_world 336 --fps_world 20 --fourcc_world MJPG `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric `
  --input_size 256 `
  --max_res 640 `
  --segment_secs 1 `
  --target_fps 10 `
  --show_fps `
  --fx_w $($W.fx) --fy_w $($W.fy) --cx_w $($W.cx) --cy_w $($W.cy) `
  --extrinsic_json .\data\extrinsic_eye_to_world.json
```
