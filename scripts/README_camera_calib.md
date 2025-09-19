# intrinsic 실행

### 캡쳐

```
# 웹캠
python scripts/calib_intrinsics.py grab ^
  --src "video=Iriun Webcam" ^
  --av_backend dshow --pixel_format mjpeg ^
  --width 640 --height 480 --fps 30 --flip 0 ^
  --save_dir calib/intrinsics/eye ^
  --chess_cols 9 --chess_rows 6


# RTSP
python calib_intrinsics.py grab \
  --src rtsp://IP:8554/cam --save_dir calib/intrinsics/eye \
  --chess_cols 9 --chess_rows 6

# 해결(캘리브레이션)
python calib_intrinsics.py solve \
  --img_glob "calib/intrinsics/world/*.jpg" \
  --chess_cols 9 --chess_rows 6 --square_m 0.025 \
  --out_json data/intrinsics_world.json \
  --write_preview calib/intrinsics/world/_undist_preview.jpg


```

# extrinsic 실행

```웹캠 2대
python scripts/estimate_extrinsic_dual_chessboard.py ^
  --eye_src "video=Iriun Webcam" ^
  --world_src "video=USB2.0 Camera" ^
  --av_backend_eye dshow --pixel_format_eye mjpeg ^
  --av_backend_world dshow --pixel_format_world mjpeg ^
  --width_eye 640 --height_eye 480 --fps_eye 30 ^
  --width_world 512 --height_world 384 --fps_world 30 ^
  --eye_intrinsic data/intrinsics_eye_640x480.json ^
  --world_intrinsic data/intrinsics_world_512x384.json ^
  --rows_a 6 --cols_a 9 --square_a 0.025 ^
  --rows_b 6 --cols_b 9 --square_b 0.025 ^
  --assume_back_to_back --thickness_m 0.050 ^
  --max_samples 30 --auto --auto_interval 0.7 ^
  --out_json data/extrinsic_eye_to_world.json

```

```RTSP 2개 스트림
python estimate_extrinsic_dual_chessboard.py \
  --eye_src rtsp://192.168.0.44:8554/cam \
  --world_src rtsp://192.168.0.68:8554/cam \
  --eye_intrinsic ./data/intrinsics_eye.json \
  --world_intrinsic ./data/intrinsics_world.json \
  --rows_a 6 --cols_a 9 --square_a 0.025 \
  --rows_b 6 --cols_b 9 --square_b 0.025 \
  --assume_back_to_back --thickness_m 0.050 \
  --max_samples 30 --auto --auto_interval 0.7 \
  --out_json ./data/extrinsic_eye_to_world.json

```

python scripts/calib_intrinsics.py grab ^
--src 1 ^
--av_backend dshow --pixel_format mjpeg ^
--width 640 --height 480 --fps 30 --flip 0 ^
--save_dir calib/intrinsics/eye ^
--chess_cols 9 --chess_rows 6

python scripts/calib_intrinsics.py grab ^
--src "video=Iriun Webcam" ^
--av_backend dshow --pixel_format mjpeg ^
--width 640 --height 480 --fps 30 --flip 0 ^
--save_dir calib/intrinsics/eye ^
--chess_cols 9 --chess_rows 6
