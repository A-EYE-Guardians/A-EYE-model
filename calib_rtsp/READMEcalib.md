# **실행 개요 (요약)**

과제 1 – Intrinsic(내부파라미터)

- RTSP 스트림에서 체스보드가 잘 보이도록 다양한 자세/거리로 이미지 캡처
- 캡처한 이미지로 calibrateCamera 실행 → intrinsic_eye.json, intrinsic_world.json 저장

과제 2 – Extrinsic(eye → world 변환)

- 두 카메라에서 동시에 체스보드를 보도록 배치 → 동시 캡처 페어를 수집

- stereoCalibrate (FIX_INTRINSIC) 로 R, T 추정 → extrinsic_eye_to_world.json 저장
  (정의: P_world = R \* P_eye + t)

# **과제 1. Intrinsic 캘리브레이션**

사용법

### 1. 캡처 (각 카메라별로 따로 수행)

```
# Eye 카메라(예: cam1)
python .\calib\intrinsics\calib_intrinsics.py grab `
  --url rtsp://192.168.162.44:8554/cam `
  --save_dir calib/intrinsics/eye `
  --chess_cols 9 --chess_rows 6

# World 카메라(예: cam2)
python .\calib\intrinsics\calib_intrinsics.py grab `
  --url rtsp://192.168.162.68:8554/cam `
  --save_dir calib/intrinsics/world `
  --chess_cols 9 --chess_rows 6

```

- 창에서 SPACE: 코너가 검출된 프레임을 저장
- E: 자동 캡처 토글(코너 검출되면 0.8초마다 자동 저장)
- ESC: 종료

### 2. 해결(solve) (각 카메라별로 따로 수행)

```
# Eye
python .\calib\intrinsics\calib_intrinsics.py solve `
  --img_glob "calib/intrinsics/eye/*.jpg" `
  --chess_cols 9 --chess_rows 6 --square_m 0.025 `
  --out_json calib/intrinsics/intrinsic_eye.json

# World
python .\calib\intrinsics\calib_intrinsics.py solve `
  --img_glob "calib/intrinsics/world/*.jpg" `
  --chess_cols 9 --chess_rows 6 --square_m 0.025 `
  --out_json calib/intrinsics/intrinsic_world.json

```

> square_m = 한 칸(정사각형) 한 변의 실제 길이(미터). 반드시 정확히 입력!

---

# **과제 2. Extrinsic 파이프라인 (eye → world, 체스보드)**

사용법

### 1. 동시 캡처(페어)

    - “백투백(back-to-back)” 가정(앞면 평행·정반대, 앞면 간격 = 지그 두께)

```
python .\calib\extrinsic\calib_extrinsic_two_boards_rtsp.py `
  --eye_url   rtsp://192.168.162.44:8554/cam `
  --world_url rtsp://192.168.162.68:8554/cam `
  --eye_intrinsic   calib/intrinsics/intrinsic_eye.json `
  --world_intrinsic calib/intrinsics/intrinsic_world.json `
  --rows_a 6 --cols_a 9 --square_a 0.025 `
  --rows_b 6 --cols_b 9 --square_b 0.025 `
  --assume_back_to_back --thickness_m 0.50 `
  --out_json extrinsic_eye_to_world.json

```

- 두 창에서 동시에 코너가 검출되면:
  - SPACE: 샘플 추가
  - E: 자동샘플(코너 움직임 생길 때 0.8초 간격)
  - BACKSPACE: 마지막 샘플 삭제
  - ENTER: 계산/저장
- 15–30 샘플 권장.

  - 거리: 대략 0.3–1.5 m 범위로 다양하게
  - 각도: yaw/pitch ±20~30° 정도, roll은 과하게 주지 않기
  - 매 샘플마다 보드 전체가 선명하게 보이도록(모션 블러, 반사광 주의)

- 결과 체크
  - 저장된 extrinsic_eye_to_world.json의 |t|(베이스라인) 이 대략 thickness와 비슷한지 확인.
  - 만약 Z축 방향이 뒤집힌 듯하면 --thickness_m의 부호 해석을 다시 점검(또는 --ba_json로 정확한 B→A R,t 제공).
