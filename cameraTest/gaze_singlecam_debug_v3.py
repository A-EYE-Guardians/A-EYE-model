#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gaze_singlecam_debug_v3.py

[핵심 목적]
- MediaPipe FaceMesh로 2D 홍채만 인식하고,
- 이를 "가상 3D 안구 모델"로 올려서 양안 시선(ray)을 구성,
- 두 레이의 교차/최근접점(또는 2D 교차 폴백)을 "화면 상"에 표시.
- 미간으로 쏠리던 문제를 해소하고, 실제 체감과 맞는 시선 이동을 제공.

[개선 포인트 (v2 -> v3)]
1) 시선 벡터 정의 변경: 각 눈의 "눈 기준점(eye center) → 홍채 중심(iris center)의 2D 오프셋"을
   yaw/pitch로 해석한 가상 3D 방향으로 시선을 정의.
2) 가상 안구 중심: 화면(z=0) 뒤쪽(-Z)으로 R_eye만큼 둠 → 레이 구성이 안정.
3) 폴백: 3D 레이가 평행/발산하면 2D 레이(eye→iris)의 교차점으로 폴백.
4) Toe-in(수렴각) 옵션: 아주 작은 yaw를 양안에 ±부여해서 교차/안정성 향상(사람 눈의 수렴 모사).
5) 튜닝 단축키: ratio(안구반지름/홍채반지름), kx/ky(민감도), EMA, CLAHE, flip, toe-in on/off 등.

[실행 예]
  python ./gaze_singlecam_debug_v3.py --cam 0 --width 640 --height 480 --fps 30 --show_fps
  (저사양) python ./cameraTest/gaze_singlecam_debug_v3.py --cam 1 --eye_stride 2 --ema 0.55 --ratio 2.0 --show_fps

단축키 요약
[ / ] : ratio(안구반지름/홍채반지름) ↓ / ↑
; / ' : EMA alpha ↓ / ↑
J / K : kx ↓ / ↑ (가로 민감도)
N / M : ky ↓ / ↑ (세로 민감도)
T : toe-in(수렴각) 토글 (0 ↔ 초기값)
C : CLAHE 토글
F : 좌우 반전 토글
Q / ESC : 종료
"""

import os, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")

import cv2
try: cv2.setNumThreads(1)
except Exception: pass

import numpy as np
import mediapipe as mp

# ====== MediaPipe FaceMesh 인덱스 ======
mp_face_mesh = mp.solutions.face_mesh
# 눈 윤곽(참고용): 미사용 가능. 필요시 표시 등에 활용.
LEFT_EYE_IDX  = [33,133,160,159,158,157,173,246,161,163,144,145,153,154,155,33]
RIGHT_EYE_IDX = [362,263,387,386,385,384,398,466,388,390,373,374,380,381,382,362]
# 홍채 4점
LEFT_IRIS_IDX  = [468,469,470,471]
RIGHT_IRIS_IDX = [473,474,475,476]
# 눈꼬리(바깥/안쪽) 인덱스: 눈 기준점 계산에 사용
LEFT_OUTER = 33
LEFT_INNER = 133
RIGHT_OUTER = 362
RIGHT_INNER = 263

# ====== 유틸 ======
class EMA:
    """벡터/스칼라/포인트 공용 EMA. 방향벡터(2D/3D)는 정규화."""
    def __init__(self, alpha: float=0.5):
        self.alpha = float(alpha); self.v = None
    def update(self, new_v):
        new_v = np.asarray(new_v, dtype=np.float32)
        if new_v.ndim == 1 and new_v.size in (2,3):
            n = np.linalg.norm(new_v)
            if n > 1e-8: new_v = new_v / n
        if self.v is None:
            self.v = new_v
        else:
            self.v = self.alpha*new_v + (1.0-self.alpha)*self.v
            if self.v.ndim == 1 and self.v.size in (2,3):
                n = np.linalg.norm(self.v)
                if n > 1e-8: self.v = self.v / n
        return self.v

def apply_clahe_bgr(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab2 = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_Lab2BGR)

def min_enclosing_circle_2d(pts2d):
    (cx,cy),r = cv2.minEnclosingCircle(pts2d.astype(np.float32))
    return float(cx), float(cy), float(r)

def eye_center_from_landmarks(lms, idx_outer, idx_inner, W, H):
    """눈 기준점: 바깥/안쪽 눈꼬리 평균"""
    p1 = np.array([lms[idx_outer].x*W, lms[idx_outer].y*H], np.float32)
    p2 = np.array([lms[idx_inner].x*W, lms[idx_inner].y*H], np.float32)
    return 0.5*(p1 + p2)

def gaze_dir_virtual(eye_center_2d, iris_center_2d, iris_radius_px, kx=1.0, ky=1.0):
    """
    2D 오프셋(eye_center -> iris_center)을 정규화하여 가상의 3D 시선 벡터 생성.
    d = normalize([kx*u, ky*v, -1])
    - u = dx / r_iris, v = dy / r_iris (무차원)
    """
    dx = (iris_center_2d[0] - eye_center_2d[0]) / max(iris_radius_px, 1e-6)
    dy = (iris_center_2d[1] - eye_center_2d[1]) / max(iris_radius_px, 1e-6)
    d = np.array([kx*dx, ky*dy, -1.0], np.float32)
    n = np.linalg.norm(d);  d = d/(n+1e-8)
    return d

def yaw_inward(d, is_left, deg=0.0):
    """양안 수렴각 모사(아주 작은 yaw 회전). deg>0: 왼쪽은 +deg, 오른쪽은 -deg."""
    if abs(deg) < 1e-6: return d
    th = np.deg2rad(deg if is_left else -deg)
    c, s = np.cos(th), np.sin(th)
    x, y, z = float(d[0]), float(d[1]), float(d[2])
    # z축 회전(Rz): 이미지 좌표상 x-오른쪽, y-아래, z-전방(-)
    x2 = c*x - s*y
    y2 = s*x + c*y
    v = np.array([x2, y2, z], np.float32)
    n = np.linalg.norm(v);  v = v/(n+1e-8)
    return v

def closest_point_between_two_rays(o1, d1, o2, d2):
    """
    두 반직선 r1(t)=o1+t*d1, r2(s)=o2+s*d2 의 최근접점 쌍(p1,p2)과 중점 pmid, 거리 dist, 평행성 지표 denom 반환
    (t,s 음수도 허용: 연장선으로 취급)
    """
    w0 = o1 - o2
    a = float(np.dot(d1,d1)); b = float(np.dot(d1,d2)); c = float(np.dot(d2,d2))
    d = float(np.dot(d1,w0));  e = float(np.dot(d2,w0))
    denom = a*c - b*b
    if abs(denom) < 1e-8:
        # 거의 평행: 한쪽으로 정사영
        t = 0.0
        s = (b*t - e) / (c + 1e-8)
    else:
        t = (b*e - c*d) / denom
        s = (a*e - b*d) / denom
    p1 = o1 + t*d1
    p2 = o2 + s*d2
    pmid = 0.5*(p1+p2)
    dist = float(np.linalg.norm(p1 - p2))
    return p1, p2, pmid, dist, denom

def line_intersection_2d(p1, p2, q1, q2):
    """
    2D 직선 p1-p2, q1-q2의 교차점 반환. 평행이면 None.
    p1,p2,q1,q2: (x, y) float32
    """
    p1 = np.asarray(p1, np.float32); p2 = np.asarray(p2, np.float32)
    q1 = np.asarray(q1, np.float32); q2 = np.asarray(q2, np.float32)
    xdiff = (p1[0]-p2[0], q1[0]-q2[0])
    ydiff = (p1[1]-p2[1], q1[1]-q2[1])
    def det(a, b): return a[0]*b[1] - a[1]*b[0]
    div = det(xdiff, ydiff)
    if abs(div) < 1e-6:
        return None
    d = (det(p1, p2), det(q1, q2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y], dtype=np.float32)

# ====== 단일 프레임 처리 ======
def gaze_step(frame_bgr, face_mesh, ema_dir_L, ema_dir_R, ema_point,
              iris_to_eyeball_ratio=2.0, kx=1.0, ky=1.0,
              toe_in_deg=0.0, flip=False, clahe=False, draw=True,
              fallback_dist_px=80.0):
    """
    반환:
      frame_vis, out(dict)
      out keys: oL,dL,cL2,rL2,oR,dR,cR2,rR2,pmid,dist,denom,pmid_ema
    """
    out = {k: None for k in ["oL","dL","cL2","rL2","oR","dR","cR2","rR2","pmid","dist","denom","pmid_ema"]}

    if flip: frame_bgr = cv2.flip(frame_bgr, 1)
    if clahe: frame_bgr = apply_clahe_bgr(frame_bgr)
    H,W = frame_bgr.shape[:2]

    # --- FaceMesh 실행 ---
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        if draw:
            cv2.putText(frame_bgr, "NO FACE", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return frame_bgr, out
    lms = res.multi_face_landmarks[0].landmark

    # ---- Left eye ----
    irisL_2d = np.array([[lms[i].x*W, lms[i].y*H] for i in LEFT_IRIS_IDX], np.float32)
    cxL, cyL, rL = min_enclosing_circle_2d(irisL_2d)
    cL2 = np.array([cxL, cyL], np.float32)
    eL2 = eye_center_from_landmarks(lms, LEFT_OUTER, LEFT_INNER, W, H)

    R_e_L = iris_to_eyeball_ratio * rL                          # 픽셀 단위 가상 안구 반지름
    oL = np.array([cxL, cyL, 0.0], np.float32) + np.array([0,0,-R_e_L], np.float32)  # 가상 3D 안구 중심
    dL_raw = gaze_dir_virtual(eL2, cL2, rL, kx=kx, ky=ky)       # 가상 3D 시선
    dL_raw = yaw_inward(dL_raw, is_left=True, deg=toe_in_deg)   # 수렴각(옵션)
    dL = ema_dir_L.update(dL_raw)
    out.update(oL=oL, dL=dL, cL2=(int(cxL),int(cyL)), rL2=rL)

    # ---- Right eye ----
    irisR_2d = np.array([[lms[i].x*W, lms[i].y*H] for i in RIGHT_IRIS_IDX], np.float32)
    cxR, cyR, rR = min_enclosing_circle_2d(irisR_2d)
    cR2 = np.array([cxR, cyR], np.float32)
    eR2 = eye_center_from_landmarks(lms, RIGHT_OUTER, RIGHT_INNER, W, H)

    R_e_R = iris_to_eyeball_ratio * rR
    oR = np.array([cxR, cyR, 0.0], np.float32) + np.array([0,0,-R_e_R], np.float32)
    dR_raw = gaze_dir_virtual(eR2, cR2, rR, kx=kx, ky=ky)
    dR_raw = yaw_inward(dR_raw, is_left=False, deg=toe_in_deg)
    dR = ema_dir_R.update(dR_raw)
    out.update(oR=oR, dR=dR, cR2=(int(cxR),int(cyR)), rR2=rR)

    # ---- Fusion: 3D 최근접점 / 평행·발산 시 2D 교차 폴백 ----
    _, _, pmid3, dist3, denom = closest_point_between_two_rays(oL, dL, oR, dR)
    out["pmid"] = pmid3; out["dist"] = dist3; out["denom"] = float(denom)

    # 기본 3D pmid의 (x,y)
    pxy = np.array([pmid3[0], pmid3[1]], np.float32)

    if (abs(denom) < 1e-6) or (dist3 > fallback_dist_px):
        # 2D 레이: base=oE 2D, tip=cE 2D 를 잇는 직선의 교차
        oL2 = np.array([oL[0], oL[1]], np.float32)
        oR2 = np.array([oR[0], oR[1]], np.float32)
        isect2d = line_intersection_2d(oL2, cL2, oR2, cR2)
        if isect2d is not None and np.isfinite(isect2d).all():
            pxy = isect2d
        else:
            # 그래도 실패시 홍채 중심 중점
            pxy = 0.5*(cL2 + cR2)

    # EMA로 화면 포인트 안정화
    pxy = ema_point.update(pxy)

    # ---- Draw ----
    if draw:
        # 홍채 원 & 중심
        cv2.circle(frame_bgr, out["cL2"], int(rL), (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, out["cL2"], 2, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, out["cR2"], int(rR), (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, out["cR2"], 2, (255,255,255), -1, cv2.LINE_AA)

        # 각 눈의 시선 화살표: base=oE(2D), tip=iris(2D) 방향으로 길이 L
        def draw_arrow_from_o_to_c(frame, o3, c2, color, L=120.0):
            base = (int(np.clip(o3[0],0,W-1)), int(np.clip(o3[1],0,H-1)))
            vec = np.array([c2[0]-base[0], c2[1]-base[1]], np.float32)
            n = np.linalg.norm(vec);  tip = base
            if n > 1e-6:
                tip = (int(base[0] + (vec[0]/n)*L), int(base[1] + (vec[1]/n)*L))
            cv2.arrowedLine(frame, base, tip, color, 2, tipLength=0.18)
        draw_arrow_from_o_to_c(frame_bgr, oL, out["cL2"], (0,255,255))
        draw_arrow_from_o_to_c(frame_bgr, oR, out["cR2"], (0,255,255))

        # 융합 포인트(교차/최근접/폴백)
        cv2.circle(frame_bgr, (int(np.clip(pxy[0],0,W-1)), int(np.clip(pxy[1],0,H-1))),
                   6, (0,0,255), -1, cv2.LINE_AA)

        # 상태 텍스트
        cv2.putText(frame_bgr,
                    f"ratio={iris_to_eyeball_ratio:.2f}  EMA={ema_dir_L.alpha:.2f}  kx={kx:.2f} ky={ky:.2f}  toe-in={toe_in_deg:.1f}",
                    (12, H-18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2, cv2.LINE_AA)

    out["pmid_ema"] = pxy
    return frame_bgr, out

# ====== 카메라 열기 ======
def open_camera(index: int, width: int, height: int, fps: float):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:    cap.set(cv2.CAP_PROP_FPS, fps)
    ok, _ = cap.read()
    return cap if ok else None

# ====== 메인 ======
def main():
    ap = argparse.ArgumentParser(description="Single-camera gaze debug (Virtual Eye Model, v3)")
    ap.add_argument("--cam", type=int, default=0, help="카메라 인덱스(정수)")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--eye_stride", type=int, default=1, help="매 N프레임마다 FaceMesh 실행")
    ap.add_argument("--ema", type=float, default=0.5, help="EMA alpha 0.3~0.7 권장")
    ap.add_argument("--ratio", type=float, default=2.0, help="iris_to_eyeball_ratio (1.8~2.3)")
    ap.add_argument("--kx", type=float, default=1.0, help="가로 민감도(iris_offset_x 스케일)")
    ap.add_argument("--ky", type=float, default=1.0, help="세로 민감도(iris_offset_y 스케일)")
    ap.add_argument("--toe_in_deg", type=float, default=1.5, help="양안 수렴각(deg, 0~3 권장)")
    ap.add_argument("--no_toe_in", action="store_true", help="toe-in 끄기")
    ap.add_argument("--clahe", action="store_true", help="대비 향상(저조도)")
    ap.add_argument("--flip", action="store_true", help="좌우 반전")
    ap.add_argument("--show_fps", action="store_true")
    ap.add_argument("--fallback_dist", type=float, default=80.0, help="3D 최소거리(px) 초과시 폴백 임계")
    args = ap.parse_args()

    print(f"[Args] cam={args.cam} size={args.width}x{args.height} fps={args.fps}")
    print(f"[Args] stride={args.eye_stride} EMA={args.ema} ratio={args.ratio} kx={args.kx} ky={args.ky} toe_in={0.0 if args.no_toe_in else args.toe_in_deg}")
    print(f"[Args] clahe={args.clahe} flip={args.flip} fallback_dist={args.fallback_dist}")

    cap = open_camera(args.cam, args.width, args.height, args.fps)
    if cap is None:
        raise SystemExit("[ERR] camera open failed")

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.3, min_tracking_confidence=0.3
    )

    ema_dir_L = EMA(args.ema)
    ema_dir_R = EMA(args.ema)
    ema_point = EMA(0.5)

    eye_idx = 0
    mesh_runs = 0
    t_last = time.time()
    mesh_hz = 0.0

    ratio = float(args.ratio)
    kx = float(args.kx)
    ky = float(args.ky)
    toe_in_deg = 0.0 if args.no_toe_in else float(args.toe_in_deg)
    use_clahe = bool(args.clahe)
    do_flip = bool(args.flip)
    fallback_dist = float(args.fallback_dist)

    print("[Main] keys: [ ] ratio  ; ' EMA  J/K kx-+  N/M ky-+  T toggle toe-in  C clahe  F flip  Q/ESC quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: continue

            run_mesh = (eye_idx % max(1, args.eye_stride) == 0)
            if run_mesh:
                t0 = time.time()
                frame_vis, out = gaze_step(
                    frame, face_mesh, ema_dir_L, ema_dir_R, ema_point,
                    iris_to_eyeball_ratio=ratio, kx=kx, ky=ky,
                    toe_in_deg=toe_in_deg, flip=do_flip, clahe=use_clahe, draw=True,
                    fallback_dist_px=fallback_dist
                )
                mesh_runs += 1
                t1 = time.time()

                if out["pmid"] is not None:
                    d = out["dist"]; denom = out["denom"]
                    print(f"[Gaze] dist={d:5.1f}px  denom={denom:.2e}  step_ms={(t1-t0)*1000:.1f}")
            else:
                frame_vis = frame.copy()

            eye_idx += 1

            # 1초 주기 FaceMesh Hz
            now = time.time()
            if now - t_last >= 1.0:
                mesh_hz = mesh_runs / max(now - t_last, 1e-3)
                mesh_runs = 0
                t_last = now
                if args.show_fps:
                    print(f"[FPS] FaceMesh ~{mesh_hz:4.1f} Hz")

            # 화면에 FaceMesh Hz 표시
            if args.show_fps:
                cv2.putText(frame_vis, f"FaceMesh Hz: {mesh_hz:4.1f}", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow("Gaze (single-cam, v3)", frame_vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q'), ord('Q')):  # ESC/Q
                break
            elif k == ord('['):
                ratio = max(1.2, ratio - 0.05)
            elif k == ord(']'):
                ratio = min(2.6, ratio + 0.05)
            elif k == ord(';'):
                ema_dir_L.alpha = ema_dir_R.alpha = max(0.1, ema_dir_L.alpha - 0.05)
            elif k == ord("'"):
                ema_dir_L.alpha = ema_dir_R.alpha = min(0.9, ema_dir_L.alpha + 0.05)
            elif k == ord('j') or k == ord('J'):
                kx = max(0.3, kx - 0.05)
            elif k == ord('k') or k == ord('K'):
                kx = min(2.5, kx + 0.05)
            elif k == ord('n') or k == ord('N'):
                ky = max(0.3, ky - 0.05)
            elif k == ord('m') or k == ord('M'):
                ky = min(2.5, ky + 0.05)
            elif k == ord('t') or k == ord('T'):
                # toe-in 토글: 0 ↔ 기본값
                toe_in_deg = 0.0 if toe_in_deg > 1e-6 else (0.0 if args.no_toe_in else float(args.toe_in_deg))
            elif k in (ord('c'), ord('C')):
                use_clahe = not use_clahe
            elif k in (ord('f'), ord('F')):
                do_flip = not do_flip

    finally:
        try: cap.release()
        except: pass
        try: face_mesh.close()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass
        print("[Main] bye.")

if __name__ == "__main__":
    main()
