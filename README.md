# 0) 가상환경 활성화

C:/Gukbi/Direct_RP_CV/venv/Scripts/Activate.ps1

- pip install -r .\requirements.txt
- pip install --no-deps ultralytics==8.3.26
- pip install --no-deps ultralytics-thop==2.0.17

# 1) intrinsics 로드

$W = Get-Content .\data\intrinsics_world.json | ConvertFrom-Json
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:KMP_AFFINITY="disabled"

# 2) 실행 (처음엔 GUI 켜서 확인 권장)

```
python .\inproc_gaze_depth_fusion_webcam.py `
  --eye_cam 1 --world_cam 0 `
  --av_backend_eye dshow --pixel_format_eye mjpeg --width_eye 320 --height_eye 240 --fps_eye 30 `
  --av_backend_world dshow --pixel_format_world mjpeg --width_world 512 --height_world 384 --fps_world 30 `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric --input_size 256 --max_res 640 --target_fps 18 `
  --device auto --show_fps `
  --fx_w $($W.fx) --fy_w $($W.fy) --cx_w $($W.cx) --cy_w $($W.cy) `
  --extrinsic_json .\data\extrinsic_eye_to_world.json
```

```
python .\ver2_inproc_gaze_depth_fusion_webcam.py `
  --eye_cam 1 --world_cam 0 `
  --av_backend_eye dshow --pixel_format_eye mjpeg --width_eye 320 --height_eye 240 --fps_eye 30 `
  --av_backend_world dshow --pixel_format_world mjpeg --width_world 512 --height_world 384 --fps_world 30 `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric --input_size 256 --max_res 640 --target_fps 18 `
  --device auto --show_fps `
  --fx_w $($W.fx) --fy_w $($W.fy) --cx_w $($W.cx) --cy_w $($W.cy) `
  --extrinsic_json .\data\extrinsic_eye_to_world.json
```

```
python .\ver3_inproc_gaze_depth_fusion_webcam.py `
  --eye_cam 1 --world_cam 0 `
  --av_backend_eye dshow --pixel_format_eye mjpeg --width_eye 320 --height_eye 240 --fps_eye 30 `
  --av_backend_world dshow --pixel_format_world mjpeg --width_world 512 --height_world 384 --fps_world 30 `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric --input_size 256 --max_res 640 --target_fps 18 `
  --device auto --show_fps --draw_rejected `
  --fx_w $($W.fx) --fy_w $($W.fy) --cx_w $($W.cx) --cy_w $($W.cy) `
  --Kref_w 512 --Kref_h 384 `
  --extrinsic_json .\data\extrinsic_eye_to_world.json
```

```
python .\ver4_inproc_gaze_depth_fusion_webcam.py `
  --eye_cam 1 --world_cam 0 `
  --av_backend_eye dshow --pixel_format_eye mjpeg --width_eye 320 --height_eye 240 --fps_eye 30 `
  --av_backend_world dshow --pixel_format_world mjpeg --width_world 512 --height_world 384 --fps_world 30 `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric --input_size 256 --max_res 640 --target_fps 18 `
  --device auto --show_fps --draw_rejected --auto_flip_dir `
  --fx_w $($W.fx) --fy_w $($W.fy) --cx_w $($W.cx) --cy_w $($W.cy) `
  --Kref_w 512 --Kref_h 384 `
  --extrinsic_json .\data\extrinsic_eye_to_world.json
```

```
python .\ver5_inproc_gaze_depth_fusion_webcam.py `
  --world_only `
  --eye_cam 1 --world_cam 0 `
  --av_backend_world dshow --pixel_format_world mjpeg ` --width_world 512 --height_world 384 --fps_world 30 `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric --input_size 256 --max_res 640 ` --target_fps 18 `
  --device auto --show_fps `
  --disable_gaze `
  --force_center `
  --fx_w $($W.fx) --fy_w $($W.fy) --cx_w $($W.cx) --cy_w $($W.cy) `
  --Kref_w 512 --Kref_h 384 `
  --extrinsic_json .\data\extrinsic_eye_to_world.json
```

```
python .\ver5_inproc_gaze_depth_fusion_webcam.py `
  --eye_cam 1 --world_cam 0 `
  --av_backend_world dshow --pixel_format_world mjpeg ` --width_world 512 --height_world 384 --fps_world 30 `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric --input_size 256 --max_res 640 ` --target_fps 18 `
  --device auto --show_fps `
  --force_center `
  --fx_w $($W.fx) --fy_w $($W.fy) --cx_w $($W.cx) --cy_w $($W.cy) `
  --Kref_w 512 --Kref_h 384 `
  --extrinsic_json .\data\extrinsic_eye_to_world.json
```

```
python .\gaze_yolo_fusion.py `
  --eye_cam 1 --world_cam 0 `
  --av_backend_world dshow --pixel_format_world mjpeg `
  --width_world 512 --height_world 384 --fps_world 30 `
  --vda_dir "C:\Gukbi\Direct_RP_CV\Video-Depth-Anything" `
  --encoder vits --metric --input_size 256 --max_res 640 `
  --target_fps 18 `
  --device auto --show_fps `
  --force_center `
  --fx_w $($W.fx) --fy_w $($W.fy) --cx_w $($W.cx) --cy_w $($W.cy) `
  --Kref_w 512 --Kref_h 384 `
  --extrinsic_json .\data\extrinsic_eye_to_world.json `
  --enable_yolo `
  --yolo_weights .\yoloe-11s-seg-pf.pt `
  --yolo_device 0 `
  --yolo_conf 0.25 --yolo_iou 0.50 --yolo_imgsz 640 `
  --yolo_outdir .\uploads `
  --yolo_unload_after_run
```

```
python .\ver2_gaze_yolo_fusion.py `
  --eye_cam 1 --world_cam 0 `
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

# 3) 하이퍼파라미터 조정 팁

파라미터 조정 가이드

1. FaceMesh 업데이트율 (원인 1)

- --eye_stride 1 : 매 프레임 실행(반응성↑, CPU↑)
- --eye_stride 2/3 : 2~3프레임마다 실행(부하↓, 반응성↓)
- 로그의 eye_run=XX/s 가 실제 FaceMesh 실행률.

2. EMA 반응성 (원인 2)

- --ema 0.5(기본): 적당히 민첩
- 더 민첩: --ema 0.6~0.8 (노이즈↑ 가능)
- 더 매끈: --ema 0.3~0.4 (지연 느낌↑)

3. 교차 유효성 필터 (원인 4)

- --res_max 0.20(기본): 잔차 20cm 이하만 유효
  - 튐이 잦으면 낮추기: 0.12~0.15
  - 너무 자주 “rejected”면 올리기: 0.25~0.35
- --r_min 0.20(기본): 레이 거리 20cm 미만은 거부
  - 실내 근거리 테스트면 0.10 정도로 완화 가능

> 콘솔의 fix: rejected (res>…, R<…) 메시지로 어떤 조건에 걸리는지 바로 확인 가능.

### 4. STT 호스트 포워더 실행

- cd C:\Gukbi\Direct_RP_CV\stt_host
- pip install websockets sounddevice (최초 한번)
- python audio_forwarder.py --ws ws://127.0.0.1:8000/stream --mic 4

“헤이 류지” → wake_detected → 말하고 멈춤 → result: ... → wake_resumed
