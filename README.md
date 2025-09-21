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
