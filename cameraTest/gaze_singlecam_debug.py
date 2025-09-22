#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gaze_singlecam_debug.py
- 카메라 1대로 MediaPipe FaceMesh(iris) 기반 시선 추정만 테스트
- 저CPU 설정(쓰레드 제한), EMA 평활화, stride로 FaceMesh 실행률 제어
- 홍채 원/방향 화살표/양안 융합 벡터 표시
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
    def __init__(self, alpha: float=0.5):
        self.alpha = float(alpha); self.v = None
    def update(self, new_v: np.ndarray):
        new_v = new_v.astype(np.float32)
        n = np.linalg.norm(new_v);  new_v = new_v/(n+1e-8)
        if self.v is None: self.v = new_v
        else:
            self.v = self.alpha*new_v + (1.0-self.alpha)*self.v
            self.v = self.v/(np.linalg.norm(self.v)+1e-8)
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

# ===== Core gaze step (단일 프레임) =====
def gaze_step(frame_bgr, face_mesh, ema_L, ema_R,
              iris_to_eyeball_ratio=2.0, flip=False, clahe=False, draw=True):
    """
    반환:
      frame_vis, out(dict)
      out keys: oL,dL,cL2,rL2,oR,dR,cR2,rR2,pmid,denom,O_eye,d_eye
    """
    out = {k: None for k in ["oL","dL","cL2","rL2","oR","dR","cR2","rR2","pmid","denom","O_eye","d_eye"]}
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

    # Left eye
    irisL_3d = lmidx_to_xyz(lms, LEFT_IRIS_IDX, W, H)
    cL3, nL = fit_plane_svd(irisL_3d); nL = orient_normal_to_camera(nL)
    irisL_2d = np.array([[lms[i].x*W, lms[i].y*H] for i in LEFT_IRIS_IDX], np.float32)
    cxL, cyL, rL = min_enclosing_circle_2d(irisL_2d)
    R_e_L = iris_to_eyeball_ratio * rL
    oL = cL3 - nL * R_e_L
    dL = ema_L.update(nL)
    out.update(oL=oL, dL=dL, cL2=(int(cxL),int(cyL)), rL2=rL)

    # Right eye
    irisR_3d = lmidx_to_xyz(lms, RIGHT_IRIS_IDX, W, H)
    cR3, nR = fit_plane_svd(irisR_3d); nR = orient_normal_to_camera(nR)
    irisR_2d = np.array([[lms[i].x*W, lms[i].y*H] for i in RIGHT_IRIS_IDX], np.float32)
    cxR, cyR, rR = min_enclosing_circle_2d(irisR_2d)
    R_e_R = iris_to_eyeball_ratio * rR
    oR = cR3 - nR * R_e_R
    dR = ema_R.update(nR)
    out.update(oR=oR, dR=dR, cR2=(int(cxR),int(cyR)), rR2=rR)

    # Cyclopean origin & direction
    w0 = oL - oR
    a = float(np.dot(dL,dL)); b = float(np.dot(dL,dR)); c = float(np.dot(dR,dR))
    d = float(np.dot(dL,w0));  e = float(np.dot(dR,w0))
    denom = a*c - b*b
    if abs(denom) < 1e-6:
        t=0.0; s = e/c if abs(c)>1e-9 else 0.0
    else:
        t = (b*e - c*d)/denom; s = (a*e - b*d)/denom
    p1 = oL + t*dL; p2 = oR + s*dR; pmid = 0.5*(p1+p2)
    out["pmid"] = pmid; out["denom"] = float(denom)

    # 최종 ray (eye 좌표 상대단위): O_eye, d_eye
    O_eye = 0.5*(oL + oR)
    d_eye = (pmid - O_eye)
    # 비등방 보정(픽셀좌표계 편향 보정)
    s_aniso = np.array([1.0, float(W)/float(H), 1.0], np.float32)
    d_eye = d_eye * s_aniso
    d_eye = d_eye/(np.linalg.norm(d_eye)+1e-8)
    out["O_eye"] = O_eye; out["d_eye"] = d_eye

    if draw:
        # 홍채 원 & 중심
        cv2.circle(frame_bgr, out["cL2"], int(rL), (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, out["cL2"], 2, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, out["cR2"], int(rR), (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, out["cR2"], 2, (255,255,255), -1, cv2.LINE_AA)

        # 각 눈의 법선 방향(화살표)
        p2L = (int(out["cL2"][0] + dL[0]*120), int(out["cL2"][1] + dL[1]*120))
        p2R = (int(out["cR2"][0] + dR[0]*120), int(out["cR2"][1] + dR[1]*120))
        cv2.arrowedLine(frame_bgr, out["cL2"], p2L, (0,255,255), 2, tipLength=0.18)
        cv2.arrowedLine(frame_bgr, out["cR2"], p2R, (0,255,255), 2, tipLength=0.18)

        # 융합 벡터(중점에서 굵은 화살표)
        base = (int((out["cL2"][0]+out["cR2"][0])/2), int((out["cL2"][1]+out["cR2"][1])/2))
        tip  = (int(base[0] + d_eye[0]*160), int(base[1] + d_eye[1]*160))
        cv2.arrowedLine(frame_bgr, base, tip, (0,0,255), 3, tipLength=0.20)

        # 텍스트 (간단 상태)
        cv2.putText(frame_bgr, f"ratio={iris_to_eyeball_ratio:.2f}  EMA={ema_L.alpha:.2f}",
                    (12, H-18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

    return frame_bgr, out

def open_camera(index: int, width: int, height: int, fps: float):
    # Windows dshow 우선 시도
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:    cap.set(cv2.CAP_PROP_FPS, fps)
    ok, frm = cap.read()
    return cap if ok else None

def main():
    ap = argparse.ArgumentParser(description="Single-camera gaze debug (FaceMesh iris)")
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
    ema_L = EMA(args.ema); ema_R = EMA(args.ema)

    eye_idx = 0
    mesh_runs = 0
    t_last = time.time()
    mesh_hz = 0.0

    ratio = float(args.ratio)
    use_clahe = bool(args.clahe)
    do_flip = bool(args.flip)

    print("[Main] running. keys: [ ] ratio  ; ' EMA  C clahe  F flip  Q/ESC quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: continue

            run_mesh = (eye_idx % max(1, args.eye_stride) == 0)
            if run_mesh:
                t0 = time.time()
                frame_vis, out = gaze_step(frame, face_mesh, ema_L, ema_R,
                                           iris_to_eyeball_ratio=ratio, flip=do_flip,
                                           clahe=use_clahe, draw=True)
                mesh_runs += 1
                t1 = time.time()

                # 디버그: 융합 벡터 값 로그
                if out["d_eye"] is not None:
                    d = out["d_eye"]; denom = out["denom"]
                    print(f"[Gaze] d=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})  denom={denom:.2e}  step_ms={(t1-t0)*1000:.1f}")
            else:
                # draw 생략, 단순 미리보기
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

            cv2.imshow("Gaze (single-cam)", frame_vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q'), ord('Q')):
                break
            elif k == ord('['):
                ratio = max(1.2, ratio - 0.05)
            elif k == ord(']'):
                ratio = min(2.6, ratio + 0.05)
            elif k == ord(';'):
                ema_L.alpha = ema_R.alpha = max(0.1, ema_L.alpha - 0.05)
            elif k == ord("'"):
                ema_L.alpha = ema_R.alpha = min(0.9, ema_L.alpha + 0.05)
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
