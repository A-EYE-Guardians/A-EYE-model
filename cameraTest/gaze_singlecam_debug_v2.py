#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gaze_singlecam_debug_v2.py
- 카메라 1대로 MediaPipe FaceMesh(iris) 기반 시선 추정만 테스트 (개선판)
- 핵심 수정: 각 눈의 '안구 중심 oE'에서 '홍채 중심 cE3'로 향하는 벡터를 시선(dE)로 사용
- 두 시선(ray) 간 최근접점/교차점(pmID) 계산하여 화면에 표시
- 저CPU 설정(쓰레드 제한), EMA 평활화(벡터/포인트), stride로 FaceMesh 실행률 제어
- 실시간 튜닝 단축키: [ ] = iris_to_eyeball_ratio, ; ' = EMA, C = CLAHE, F = flip
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

# ===== FaceMesh indices =====
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDX  = [33,133,160,159,158,157,173,246,161,163,144,145,153,154,155,33]
RIGHT_EYE_IDX = [362,263,387,386,385,384,398,466,388,390,373,374,380,381,382,362]
LEFT_IRIS_IDX  = [468,469,470,471]
RIGHT_IRIS_IDX = [473,474,475,476]

# ===== Utils =====
class EMA:
    """벡터/스칼라/좌표 모두 가능(넘겨준 타입 그대로 업데이트)"""
    def __init__(self, alpha: float=0.5):
        self.alpha = float(alpha); self.v = None
    def update(self, new_v):
        new_v = np.asarray(new_v, dtype=np.float32)
        # 벡터이면 단위화 옵션: 방향벡터 안정화를 위해 사용
        if new_v.ndim == 1 and new_v.size in (2,3):
            n = np.linalg.norm(new_v)
            if n > 1e-8:
                new_v = new_v / n
        if self.v is None:
            self.v = new_v
        else:
            self.v = self.alpha * new_v + (1.0 - self.alpha) * self.v
            # 방향벡터면 다시 정규화
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

def lm3d_px(lm, W, H): return np.array([lm.x*W, lm.y*H, lm.z*W], np.float32)
def lmidx_to_xyz(landmarks, idx_list, W, H): return np.array([lm3d_px(landmarks[i], W, H) for i in idx_list], np.float32)

def fit_plane_svd(pts3d):
    """평면 중심 c, 법선 n (단위)"""
    c = pts3d.mean(0); X = pts3d - c
    _,_,vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1]; n = n/(np.linalg.norm(n)+1e-8); return c, n

def orient_normal_to_camera(n):
    # 카메라 전방을 -Z로 가정(미디어파이프 좌표 관례 기준)
    if np.dot(n, np.array([0,0,-1], np.float32)) < 0: n = -n
    return n

def min_enclosing_circle_2d(pts2d):
    (cx,cy),r = cv2.minEnclosingCircle(pts2d.astype(np.float32))
    return float(cx), float(cy), float(r)

def closest_point_between_two_rays(o1, d1, o2, d2):
    """
    두 반직선 r1(t)=o1+t*d1, r2(s)=o2+s*d2 의 최근접점 쌍(p1,p2)과 중점 pmid, 거리 dist, 평행성 지표 denom 반환
    (t,s는 음수가 될 수도 있으나, 시선 연장선으로 취급)
    """
    w0 = o1 - o2
    a = float(np.dot(d1,d1)); b = float(np.dot(d1,d2)); c = float(np.dot(d2,d2))
    d = float(np.dot(d1,w0));  e = float(np.dot(d2,w0))
    denom = a*c - b*b
    if abs(denom) < 1e-8:
        # 거의 평행: 한 쪽에 정사영
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

# ===== Core gaze step (단일 프레임) =====
def gaze_step(frame_bgr, face_mesh, ema_dir_L, ema_dir_R, ema_point,
              iris_to_eyeball_ratio=2.0, flip=False, clahe=False, draw=True,
              intersect_tol_px=3.0, fallback_dist_px=60.0):
    """
    반환:
      frame_vis, out(dict)
      out keys:
        oL,dL,cL2,rL2,oR,dR,cR2,rR2,pmid,dist,denom,pmid_ema
    """
    out = {k: None for k in ["oL","dL","cL2","rL2","oR","dR","cR2","rR2","pmid","dist","denom","pmid_ema"]}
    if flip: frame_bgr = cv2.flip(frame_bgr, 1)
    if clahe: frame_bgr = apply_clahe_bgr(frame_bgr)
    H,W = frame_bgr.shape[:2]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        if draw:
            cv2.putText(frame_bgr, "NO FACE", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return frame_bgr, out

    lms = res.multi_face_landmarks[0].landmark

    # ---- Left eye ----
    irisL_3d = lmidx_to_xyz(lms, LEFT_IRIS_IDX, W, H)
    cL3, nL = fit_plane_svd(irisL_3d); nL = orient_normal_to_camera(nL)
    irisL_2d = np.array([[lms[i].x*W, lms[i].y*H] for i in LEFT_IRIS_IDX], np.float32)
    cxL, cyL, rL = min_enclosing_circle_2d(irisL_2d)
    R_e_L = iris_to_eyeball_ratio * rL                             # 픽셀 기반 상대 반지름
    oL = cL3 - nL * R_e_L                                          # 안구 중심(픽셀 좌표계 상대 단위)
    dL_raw = cL3 - oL                                              # "안구 중심→홍채 중심" 방향 = 시선
    dL = ema_dir_L.update(dL_raw)                                  # 방향 EMA
    out.update(oL=oL, dL=dL, cL2=(int(cxL),int(cyL)), rL2=rL)

    # ---- Right eye ----
    irisR_3d = lmidx_to_xyz(lms, RIGHT_IRIS_IDX, W, H)
    cR3, nR = fit_plane_svd(irisR_3d); nR = orient_normal_to_camera(nR)
    irisR_2d = np.array([[lms[i].x*W, lms[i].y*H] for i in RIGHT_IRIS_IDX], np.float32)
    cxR, cyR, rR = min_enclosing_circle_2d(irisR_2d)
    R_e_R = iris_to_eyeball_ratio * rR
    oR = cR3 - nR * R_e_R
    dR_raw = cR3 - oR
    dR = ema_dir_R.update(dR_raw)
    out.update(oR=oR, dR=dR, cR2=(int(cxR),int(cyR)), rR2=rR)

    # ---- Fusion: 두 시선의 최근접점/교차점 ----
    _, _, pmid, dist, denom = closest_point_between_two_rays(oL, dL, oR, dR)
    out["pmid"] = pmid; out["dist"] = dist; out["denom"] = float(denom)

    # 화면에 찍을 2D 포인트 (x,y만 사용)
    pxy = np.array([pmid[0], pmid[1]], np.float32)

    # 안정화: 두 레이 간 거리가 너무 크거나 평행이면 이전 포인트 EMA로 완화
    if (abs(denom) < 1e-6) or (dist > fallback_dist_px):
        pxy = ema_point.update(pxy)
    else:
        pxy = ema_point.update(pxy)  # 정상이어도 약간의 EMA로 부드럽게

    # ---- Draw ----
    if draw:
        # 홍채 원 & 중심
        cv2.circle(frame_bgr, out["cL2"], int(rL), (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, out["cL2"], 2, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, out["cR2"], int(rR), (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, out["cR2"], 2, (255,255,255), -1, cv2.LINE_AA)

        # 각 눈의 시선 화살표: base=oE의 2D, tip= cE2D로 향하게 (scale로 길이 조정)
        def draw_arrow_from_o_to_c(frame, o3, c2, color):
            base = (int(np.clip(o3[0],0,W-1)), int(np.clip(o3[1],0,H-1)))
            vec = np.array([c2[0]-base[0], c2[1]-base[1]], np.float32)
            n = np.linalg.norm(vec);  L = 120.0
            tip = (int(base[0] + (vec[0]/(n+1e-8))*L), int(base[1] + (vec[1]/(n+1e-8))*L))
            cv2.arrowedLine(frame, base, tip, color, 2, tipLength=0.18)

        draw_arrow_from_o_to_c(frame_bgr, oL, out["cL2"], (0,255,255))
        draw_arrow_from_o_to_c(frame_bgr, oR, out["cR2"], (0,255,255))

        # 융합 포인트(교차/최근접)
        cv2.circle(frame_bgr, (int(np.clip(pxy[0],0,W-1)), int(np.clip(pxy[1],0,H-1))),
                   6, (0,0,255), -1, cv2.LINE_AA)

        # 텍스트 (상태)
        cv2.putText(frame_bgr, f"ratio={iris_to_eyeball_ratio:.2f}  EMA={ema_dir_L.alpha:.2f}  dist={dist:5.1f}px",
                    (12, H-18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

    out["pmid_ema"] = pxy
    return frame_bgr, out

def open_camera(index: int, width: int, height: int, fps: float):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:    cap.set(cv2.CAP_PROP_FPS, fps)
    ok, _ = cap.read()
    return cap if ok else None

def main():
    ap = argparse.ArgumentParser(description="Single-camera gaze debug (FaceMesh iris, v2)")
    ap.add_argument("--cam", type=int, default=0, help="카메라 인덱스(정수)")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--eye_stride", type=int, default=1, help="매 N프레임마다 FaceMesh 실행")
    ap.add_argument("--ema", type=float, default=0.5, help="EMA alpha 0.3~0.7 권장")
    ap.add_argument("--ratio", type=float, default=2.0, help="iris_to_eyeball_ratio (1.8~2.2)")
    ap.add_argument("--clahe", action="store_true", help="대비 향상(저조도)")
    ap.add_argument("--flip", action="store_true", help="좌우 반전")
    ap.add_argument("--show_fps", action="store_true")
    ap.add_argument("--fallback_dist", type=float, default=60.0, help="두 시선 최소거리(px) 초과시 포인트 EMA로 완화")
    args = ap.parse_args()

    print(f"[Args] cam={args.cam} size={args.width}x{args.height} fps={args.fps}")
    print(f"[Args] stride={args.eye_stride} EMA={args.ema} ratio={args.ratio} clahe={args.clahe} flip={args.flip}")

    cap = open_camera(args.cam, args.width, args.height, args.fps)
    if cap is None:
        raise SystemExit("[ERR] camera open failed")

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.3, min_tracking_confidence=0.3
    )

    # 방향/포인트 별 EMA를 분리
    ema_dir_L = EMA(args.ema)
    ema_dir_R = EMA(args.ema)
    ema_point = EMA(0.5)  # 화면 표시 포인트는 방향보다 빠르게도/느리게도 가능. 필요시 키로 조절해도 됨.

    eye_idx = 0
    mesh_runs = 0
    t_last = time.time()
    mesh_hz = 0.0

    ratio = float(args.ratio)
    use_clahe = bool(args.clahe)
    do_flip = bool(args.flip)
    fallback_dist = float(args.fallback_dist)

    print("[Main] keys: [ ] ratio  ; ' EMA  C clahe  F flip  Q/ESC quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: continue

            run_mesh = (eye_idx % max(1, args.eye_stride) == 0)
            if run_mesh:
                t0 = time.time()
                frame_vis, out = gaze_step(
                    frame, face_mesh, ema_dir_L, ema_dir_R, ema_point,
                    iris_to_eyeball_ratio=ratio, flip=do_flip,
                    clahe=use_clahe, draw=True,
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

            # 1초 주기 FPS
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

            cv2.imshow("Gaze (single-cam, v2)", frame_vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q'), ord('Q')):
                break
            elif k == ord('['):
                ratio = max(1.2, ratio - 0.05)
            elif k == ord(']'):
                ratio = min(2.6, ratio + 0.05)
            elif k == ord(';'):
                ema_dir_L.alpha = ema_dir_R.alpha = max(0.1, ema_dir_L.alpha - 0.05)
            elif k == ord("'"):
                ema_dir_L.alpha = ema_dir_R.alpha = min(0.9, ema_dir_L.alpha + 0.05)
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
        print("[Main] bye.]")

if __name__ == "__main__":
    main()