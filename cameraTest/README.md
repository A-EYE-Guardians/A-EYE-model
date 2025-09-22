```
# (가상환경 활성화했다는 가정)
python .\cameraTest\gaze_singlecam_debug.py `
  --cam 1 `
  --width 320 --height 240 --fps 30 `
  --eye_stride 1 `
  --ema 0.5 `
  --ratio 2.0 `
  --show_fps

```

--cam: 테스트할 카메라 인덱스(숫자). (네 PC에선 보통 0 또는 1)
--width/--height/--fps: 캡처 설정
--eye_stride: FaceMesh 실행 간격(1=매 프레임, 2=격프레임)
--ema: 시선 방향 평활화(0.3~0.7 권장)
--ratio: 홍채반지름→안구반지름 비율(보통 1.8~2.2, 기본 2.0)
--clahe: 저조도/노이즈에서 대비 향상 쓰고 싶으면 옵션으로 추가
--flip: 좌/우 반전

런타임 단축키

[ / ] : iris_to_eyeball_ratio 즉석 조정 (작게 → 시선 화살표가 더 바깥으로, 크게 → 더 안쪽)
; / ' : EMA 알파 조정(반응성 vs 안정성)
C : CLAHE 토글
F : 좌우 반전 토글
Q / ESC : 종료

python .\cameraTest\gaze_singlecam_debug_v2.py --cam 1 --width 320 --height 240 --fps 30 --eye_stride 2 --ema 0.5 --ratio 2.0 --show_fps



python .\cameraTest\gaze_singlecam_debug_v3_1.py --cam 1 --width 640 --height 480 --fps 30 `
  --canvas --canvas_width 640 --canvas_height 480 `
  --toe_in_deg 2.3 --kx 0.95 --ky 1.35 --fallback_dist 140 --ema 0.6 --ratio 1.95 --show_fps

