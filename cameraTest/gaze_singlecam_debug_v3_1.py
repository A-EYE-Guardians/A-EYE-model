#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gaze_singlecam_debug_v3_1.py

[핵심 기능]
- MediaPipe FaceMesh로 2D 홍채 검출
- 가상 3D 안구 모델로 양안 시선(ray) 구성
- 3D 최근접점(교차점 유사) + 2D 교차 폴백
- 안정화 패치:
  A) 평행+과도 이격 아웃라이어 프레임 스킵
  B) 2D 폴백은 (안구중심→홍채중심) 대신 (눈기준점→홍채중심) 직선 교차 사용
- 디버그 로그: cos(dL,dR), IPD(px), dist, denom
- 추가: "검은색 캔버스" 창에 현재 시점을 십자가로 표시

[실행 예]
  python ./gaze_singlecam_debug_v3_1.py --cam 0 --width 640 --height 480 --fps 30 --show_fps --canvas --toe_in_deg 2.3 --kx 0.95 --ky 1.35 --fallback_dist 140 --ema 0.6 --ratio 1.95
"""

import os, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")

import cv2
try:
    cv2.setNumThreads(1)
except Exception:
    pass

import numpy as np
import mediapipe as mp

# ====== MediaPipe FaceMesh 인덱스 ======
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDX  = [33,133,160,159,158,157,173,246,161,163,144,145,153,154,155,33]
RIGHT_EYE_IDX = [362,263,387,386,385,384,398,466,388,390,373,374,380,381,382,362]
LEFT_IRIS_IDX  = [468,469,470,471]
RIGHT_IRIS_IDX = [473,474,475,476]

LEFT_OUTER = 33
LEFT_INNER = 133
RIGHT_OUTER = 362
RIGHT_INNER = 263

# ====== 유틸 ======
class EMA:
    """벡터/스칼라/포인트 공용 EMA. 방향벡터(2D/3D)는 정규화."""
    def __init__(self, alpha: float=0.5):
        self.alpha = float(alpha)
        self.v = None
    def update(self, new_v):
        new_v = np.asarray(new_v, dtype=np.float32)
        if new_v.ndim == 1 and new_v.size in (2,3):
            n = np.linalg.norm(new_v)
            if n > 1e-8:
                new_v = new_v / n
        if self.v is None:
            self.v = new_v
        else:
            self.v = self.alpha*new_v + (1.0-self.alpha)*self.v
            if self.v.ndim == 1 and self.v.size in (2,3):
                n = np.linalg.norm(self.v)
                if n > 1e-8:
                    self.v = self.v / n
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
    - u = dx / r_iris, v = dy / r_iris
    """
    dx = (iris_center_2d[0] - eye_center_2d[0]) / max(iris_radius_px, 1e-6)
    dy = (iris_center_2d[1] - eye_center_2d[1]) / max(iris_radius_px, 1e-6)
    d = np.array([kx*dx, ky*dy, -1.0], np.float32)
    n = np.linalg.norm(d)
    d = d/(n+1e-8)
    return d

def yaw_inward(d, is_left, deg=0.0):
    """양안 수렴각 모사(아주 작은 yaw 회전). deg>0: 왼쪽은 +deg, 오른쪽은 -deg."""
    if abs(deg) < 1e-6:
        return d
    th = np.deg2rad(deg if is_left else -deg)
    c, s = np.cos(th), np.sin(th)
    x, y, z = float(d[0]), float(d[1]), float(d[2])
    # z축 회전(Rz): 이미지 좌표상 x-오른쪽(+), y-아래(+), z-전방(-)
    x2 = c*x - s*y
    y2 = s*x + c*y
    v = np.array([x2, y2, z], np.float32)
    n = np.linalg.norm(v)
    v = v/(n+1e-8)
    return v

def closest_point_between_two_rays(o1, d1, o2, d2):
    """
    두 반직선 r1(t)=o1+t*d1, r2(s)=o2+s*d2 의 최근접점 쌍(p1,p2)과 중점 pmid, 거리 dist, 평행성 지표 denom 반환
    (t,s 음수도 허용: 연장선 취급)
    """
    w0 = o1 - o2
    a = float(np.dot(d1,d1)); b = float(np.dot(d1,d2)); c = float(np.dot(d2,d2))
    d = float(np.dot(d1,w0));  e = float(np.dot(d2,w0))
    denom = a*c - b*b
    if abs(denom) < 1e-8:
        # 거의 평행: 한쪽 정사영
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

def draw_cross(canvas, x, y, size=12, thickness=2, color=(255,255,255)):
    """검은 캔버스에 십자가(+) 그리기"""
    h, w = canvas.shape[:2]
    xi = int(np.clip(x, 0, w-1)); yi = int(np.clip(y, 0, h-1))
    cv2.line(canvas, (max(0, xi-size), yi), (min(w-1, xi+size), yi), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (xi, max(0, yi-size)), (xi, min(h-1, yi+size)), color, thickness, cv2.LINE_AA)

# ====== 단일 프레임 처리 ======
def gaze_step(frame_bgr, face_mesh, ema_dir_L, ema_dir_R, ema_point,
              iris_to_eyeball_ratio=2.0, kx=1.0, ky=1.0,
              toe_in_deg=0.0, flip=False, clahe=False, draw=True,
              fallback_dist_px=120.0):
    """
    반환:
      frame_vis, out(dict)
      out keys: oL,dL,cL2,rL2,oR,dR,cR2,rR2,pmid,dist,denom,pmid_ema,eL2,eR2
    """
    out = {k: None for k in ["oL","dL","cL2","rL2","oR","dR","cR2","rR2","pmid","dist","denom","pmid_ema","eL2","eR2"]}

    if flip: frame_bgr = cv2.flip(frame_bgr, 1)
    if clahe: frame_bgr = apply_clahe_bgr(frame_bgr)
    H, W = frame_bgr.shape[:2]

    # FaceMesh
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        if draw:
            cv2.putText(frame_bgr, "NO FACE", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return frame_bgr, out
    lms = res.multi_face_landmarks[0].landmark

    # ---- Left ----
    irisL_2d = np.array([[lms[i].x*W, lms[i].y*H] for i in LEFT_IRIS_IDX], np.float32)
    cxL, cyL, rL = min_enclosing_circle_2d(irisL_2d)
    cL2 = np.array([cxL, cyL], np.float32)
    eL2 = eye_center_from_landmarks(lms, LEFT_OUTER, LEFT_INNER, W, H)

    R_e_L = iris_to_eyeball_ratio * rL
    oL = np.array([cxL, cyL, 0.0], np.float32) + np.array([0,0,-R_e_L], np.float32)
    dL_raw = gaze_dir_virtual(eL2, cL2, rL, kx=kx, ky=ky)
    dL_raw = yaw_inward(dL_raw, is_left=True, deg=toe_in_deg)
    dL = ema_dir_L.update(dL_raw)
    out.update(oL=oL, dL=dL, cL2=(int(cxL),int(cyL)), rL2=rL, eL2=eL2)

    # ---- Right ----
    irisR_2d = np.array([[lms[i].x*W, lms[i].y*H] for i in RIGHT_IRIS_IDX], np.float32)
    cxR, cyR, rR = min_enclosing_circle_2d(irisR_2d)
    cR2 = np.array([cxR, cyR], np.float32)
    eR2 = eye_center_from_landmarks(lms, RIGHT_OUTER, RIGHT_INNER, W, H)

    R_e_R = iris_to_eyeball_ratio * rR
    oR = np.array([cxR, cyR, 0.0], np.float32) + np.array([0,0,-R_e_R], np.float32)
    dR_raw = gaze_dir_virtual(eR2, cR2, rR, kx=kx, ky=ky)
    dR_raw = yaw_inward(dR_raw, is_left=False, deg=toe_in_deg)
    dR = ema_dir_R.update(dR_raw)
    out.update(oR=oR, dR=dR, cR2=(int(cxR),int(cyR)), rR2=rR, eR2=eR2)

    # ---- 3D Fusion ----
    _, _, pmid3, dist3, denom = closest_point_between_two_rays(oL, dL, oR, dR)
    out["pmid"] = pmid3; out["dist"] = dist3; out["denom"] = float(denom)

    # 디버그: cos(dL,dR), IPD(px)
    cos_lr = float(np.clip(np.dot(dL, dR), -1.0, 1.0))
    ipd_px = float(np.linalg.norm(eL2 - eR2))

    # 기본 3D pmid의 (x,y)
    pxy = np.array([pmid3[0], pmid3[1]], np.float32)

    # === 안정화 패치 A: 아웃라이어 스킵 ===
    par_parallel = (abs(denom) < 1e-3)
    par_far = (dist3 > 1.5 * fallback_dist_px)
    if par_parallel and par_far and ema_point.v is not None:
        # 이전 포인트 유지 + 관성만 반영
        pxy = ema_point.v.copy()
        pxy = ema_point.update(pxy)
    else:
        # === 폴백(패치 B 적용): 2D 교차는 (눈기준점→홍채중심) 직선을 사용 ===
        if (abs(denom) < 1e-6) or (dist3 > fallback_dist_px):
            isect2d = line_intersection_2d(eL2, cL2, eR2, cR2)
            if isect2d is not None and np.isfinite(isect2d).all():
                pxy = isect2d
            else:
                pxy = 0.5*(cL2 + cR2)
        # EMA로 화면 포인트 안정화
        pxy = ema_point.update(pxy)

    out["pmid_ema"] = pxy

    # ---- Draw ----
    if draw:
        # 홍채
        cv2.circle(frame_bgr, (int(cxL),int(cyL)), int(rL), (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, (int(cxR),int(cyR)), int(rR), (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, (int(cxL),int(cyL)), 2, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, (int(cxR),int(cyR)), 2, (255,255,255), -1, cv2.LINE_AA)

        # 각 눈 화살표: base=oE(2D), tip=iris(2D) 방향으로 길이 L
        def draw_arrow_from_o_to_c(frame, o3, c2, color, L=120.0):
            base = (int(np.clip(o3[0],0,W-1)), int(np.clip(o3[1],0,H-1)))
            vec = np.array([c2[0]-base[0], c2[1]-base[1]], np.float32)
            n = np.linalg.norm(vec);  tip = base
            if n > 1e-6:
                tip = (int(base[0] + (vec[0]/n)*L), int(base[1] + (vec[1]/n)*L))
            cv2.arrowedLine(frame, base, tip, color, 2, tipLength=0.18)
        draw_arrow_from_o_to_c(frame_bgr, oL, (int(cxL),int(cyL)), (0,255,255))
        draw_arrow_from_o_to_c(frame_bgr, oR, (int(cxR),int(cyR)), (0,255,255))

        # 융합 포인트
        cv2.circle(frame_bgr, (int(np.clip(pxy[0],0,W-1)), int(np.clip(pxy[1],0,H-1))),
                   6, (0,0,255), -1, cv2.LINE_AA)

        # 상태 텍스트
        cv2.putText(frame_bgr,
                    f"ratio={iris_to_eyeball_ratio:.2f} EMA={ema_dir_L.alpha:.2f} kx={kx:.2f} ky={ky:.2f} toe-in={toe_in_deg:.1f} cos={cos_lr:.3f} ipd={ipd_px:.1f}px",
                    (12, H-18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255,255,255), 2, cv2.LINE_AA)

    # 콘솔 로그(간결)
    print(f"[Gaze] dist={dist3:6.1f}px  denom={denom:.2e}  cos={cos_lr:.3f}  ipd={ipd_px:5.1f}px")

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
    ap = argparse.ArgumentParser(description="Single-camera gaze debug (Virtual Eye Model, v3.1)")
    ap.add_argument("--cam", type=int, default=0, help="카메라 인덱스(정수)")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--eye_stride", type=int, default=1, help="매 N프레임마다 FaceMesh 실행")
    ap.add_argument("--ema", type=float, default=0.6, help="EMA alpha 0.3~0.7 권장")
    ap.add_argument("--ratio", type=float, default=1.95, help="iris_to_eyeball_ratio (1.8~2.3)")
    ap.add_argument("--kx", type=float, default=0.95, help="가로 민감도(iris_offset_x 스케일)")
    ap.add_argument("--ky", type=float, default=1.35, help="세로 민감도(iris_offset_y 스케일)")
    ap.add_argument("--toe_in_deg", type=float, default=2.3, help="양안 수렴각(deg, 0~3 권장)")
    ap.add_argument("--no_toe_in", action="store_true", help="toe-in 끄기")
    ap.add_argument("--clahe", action="store_true", help="대비 향상(저조도)")
    ap.add_argument("--flip", action="store_true", help="좌우 반전")
    ap.add_argument("--show_fps", action="store_true")
    ap.add_argument("--fallback_dist", type=float, default=140.0, help="3D 최소거리(px) 초과시 2D 폴백 임계")
    # 캔버스 옵션
    ap.add_argument("--canvas", action="store_true", help="검은 캔버스 창 띄우기")
    ap.add_argument("--canvas_width", type=int, default=640)
    ap.add_argument("--canvas_height", type=int, default=480)
    args = ap.parse_args()

    print(f"[Args] cam={args.cam} size={args.width}x{args.height} fps={args.fps}")
    print(f"[Args] stride={args.eye_stride} EMA={args.ema} ratio={args.ratio} kx={args.kx} ky={args.ky} toe_in={0.0 if args.no_toe_in else args.toe_in_deg}")
    print(f"[Args] clahe={args.clahe} flip={args.flip} fallback_dist={args.fallback_dist} canvas={args.canvas} ({args.canvas_width}x{args.canvas_height})")

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
            if not ok:
                continue

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
            else:
                frame_vis = frame.copy()

            eye_idx += 1

            # FaceMesh Hz 표시
            now = time.time()
            if now - t_last >= 1.0:
                mesh_hz = mesh_runs / max(now - t_last, 1e-3)
                mesh_runs = 0
                t_last = now
                if args.show_fps:
                    print(f"[FPS] FaceMesh ~{mesh_hz:4.1f} Hz")

            # 오버레이 FPS
            if args.show_fps:
                cv2.putText(frame_vis, f"FaceMesh Hz: {mesh_hz:4.1f}", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # --- 가상 검은 캔버스에 십자가 표시 ---
            if args.canvas and (out["pmid_ema"] is not None):
                Hc, Wc = int(args.canvas_height), int(args.canvas_width)
                canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)

                # 실제 프레임 크기 사용 (args.width/height 말고)
                Hf, Wf = frame_vis.shape[:2]
                sx = float(Wc) / max(1.0, float(Wf))
                sy = float(Hc) / max(1.0, float(Hf))

                px = float(out["pmid_ema"][0]) * sx
                py = float(out["pmid_ema"][1]) * sy

                draw_cross(canvas, px, py, size=14, thickness=2, color=(255,255,255))
                cv2.imshow("Gaze Canvas", canvas)

            cv2.imshow("Gaze (single-cam, v3.1)", frame_vis)
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
            elif k in (ord('j'), ord('J')):
                kx = max(0.3, kx - 0.05)
            elif k in (ord('k'), ord('K')):
                kx = min(2.5, kx + 0.05)
            elif k in (ord('n'), ord('N')):
                ky = max(0.3, ky - 0.05)
            elif k in (ord('m'), ord('M')):
                ky = min(2.5, ky + 0.05)
            elif k in (ord('t'), ord('T')):
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
