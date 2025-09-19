#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
두 보드 지그(보드A=월드, 보드B=아이)를 이용해 eye->world extrinsic(R,t) 추정.
- RTSP 입력(라즈베리파이 MediaMTX) 실시간 샘플링
- 보드A/B는 서로 등/상이거나(행/열·스퀘어 크기 동일해도 되고 달라도 됨)
- 백투백 가정 또는 보드B->보드A의 R,t(JSON) 제공 방식 지원
- 각 샘플에서 solvePnP로 (boardA->worldCam), (boardB->eyeCam) 추정
- 체인: eye->world = (world<-A) o (A<-B) o (B<-eye)
- 수집된 여러 샘플을 RANSAC-유사 필터(각도/변위) 후
  회전=사원수 평균, 병진=중앙값으로 합성
- 출력: extrinsic_eye_to_world.json  (R row-major 9개, t 3개)
"""

import os, json, time, math, argparse
from typing import List, Tuple

import cv2
import numpy as np

# ---------- 기본 유틸 ----------
def to_K(fx, fy, cx, cy):
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)

def load_intrinsic(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    K = np.array(meta["K"], dtype=np.float64).reshape(3,3)
    dist = np.array(meta["dist"], dtype=np.float64).reshape(-1,1)
    return K, dist, tuple(meta["image_size"])

def build_objp(rows:int, cols:int, square_m:float):
    objp = np.zeros((rows*cols,3), np.float32)
    grid = np.mgrid[0:cols,0:rows].T.reshape(-1,2)
    objp[:,:2] = grid * square_m
    return objp

def find_corners(img, rows, cols):
    """체스보드 내부 코너 검출(SB 우선, 실패시 일반 알고리즘)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    patt = (cols, rows)
    ok, corners = False, None
    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            ok, corners = cv2.findChessboardCornersSB(gray, patt, flags=flags)
        except Exception:
            ok = False
    if not ok:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, patt, flags=flags)
        if ok:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term)
    return ok, corners

def rodrigues(R=None, rvec=None):
    if R is not None:
        rvec,_ = cv2.Rodrigues(R); return rvec
    if rvec is not None:
        R,_ = cv2.Rodrigues(rvec); return R
    raise ValueError

def invert_RT(R, t):
    Rt = R.T; tt = -Rt @ t
    return Rt, tt

def compose(R2,t2,R1,t1):
    """(R2,t2) o (R1,t1) : X' = R2*(R1*X + t1) + t2"""
    R = R2 @ R1
    t = R2 @ t1 + t2
    return R,t

def R_to_quat(R):
    """행렬→사원수(x,y,z,w)"""
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr+1.0)*2
        q[3]=0.25*S
        q[0]=(R[2,1]-R[1,2])/S
        q[1]=(R[0,2]-R[2,0])/S
        q[2]=(R[1,0]-R[0,1])/S
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i==0:
            S = math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
            q[3]=(R[2,1]-R[1,2])/S; q[0]=0.25*S
            q[1]=(R[0,1]+R[1,0])/S; q[2]=(R[0,2]+R[2,0])/S
        elif i==1:
            S = math.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2
            q[3]=(R[0,2]-R[2,0])/S
            q[0]=(R[0,1]+R[1,0])/S; q[1]=0.25*S; q[2]=(R[1,2]+R[2,1])/S
        else:
            S = math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2
            q[3]=(R[1,0]-R[0,1])/S
            q[0]=(R[0,2]+R[2,0])/S; q[1]=(R[1,2]+R[2,1])/S; q[2]=0.25*S
    return q

def quat_to_R(q):
    x,y,z,w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)

def mean_rotation_quat(Rs: List[np.ndarray]) -> np.ndarray:
    qs=[]
    for R in Rs:
        q=R_to_quat(R)
        if len(qs)>0 and np.dot(qs[0], q)<0: q=-q   # 부호 정렬
        qs.append(q)
    q = np.mean(np.stack(qs,0),0)
    q /= np.linalg.norm(q)
    return quat_to_R(q)

def rot_angle_deg(R):
    """회전행렬 각도(도)"""
    tr = (np.trace(R)-1)/2
    tr = min(1.0, max(-1.0, tr))
    return math.degrees(math.acos(tr))

# ---------- 메인 파이프라인 ----------
def main():
    ap = argparse.ArgumentParser()
    # RTSP
    ap.add_argument("--eye_url", type=str, required=True)
    ap.add_argument("--world_url", type=str, required=True)
    # Intrinsics JSON
    ap.add_argument("--eye_intrinsic", type=str, required=True)
    ap.add_argument("--world_intrinsic", type=str, required=True)
    # Board A(월드), Board B(아이)
    ap.add_argument("--rows_a", type=int, required=True)
    ap.add_argument("--cols_a", type=int, required=True)
    ap.add_argument("--square_a", type=float, required=True)
    ap.add_argument("--rows_b", type=int, required=True)
    ap.add_argument("--cols_b", type=int, required=True)
    ap.add_argument("--square_b", type=float, required=True)
    # B->A 관계
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--assume_back_to_back", action="store_true",
                   help="보드 앞면 평행·정반대 + 앞면 간격 thickness_m")
    ap.add_argument("--thickness_m", type=float, default=0.0,
                    help="앞면-앞면 간격(m), back-to-back일 때 사용")
    g.add_argument("--ba_json", type=str, help="보드B→보드A 변환 JSON({R:[...9], t:[3]})")
    # 샘플링/필터
    ap.add_argument("--max_samples", type=int, default=40)
    ap.add_argument("--auto", action="store_true", help="자동샘플 시작")
    ap.add_argument("--auto_interval", type=float, default=0.8)
    ap.add_argument("--min_motion_px", type=float, default=15.0,
                    help="자동샘플 시 프레임간 코너 이동 최소량")
    ap.add_argument("--reproj_thresh_px", type=float, default=1.5,
                    help="solvePnP 후 재투영 오차 임계(픽셀). 초과 샘플 폐기")
    ap.add_argument("--angle_inlier_deg", type=float, default=5.0,
                    help="최종 RANSAC 유사 각도 임계(도)")
    ap.add_argument("--trans_inlier_m", type=float, default=0.05,
                    help="최종 RANSAC 유사 변위 임계(미터)")
    # 출력
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    # Intrinsics 로드
    K_e, dist_e, _ = load_intrinsic(args.eye_intrinsic)
    K_w, dist_w, _ = load_intrinsic(args.world_intrinsic)

    # 보드B→보드A
    if args.assume_back_to_back:
        R_ab = np.diag([-1.0, 1.0, -1.0])               # Ry(pi)
        t_ab = np.array([0.0, 0.0, -args.thickness_m], dtype=np.float64).reshape(3,1)
    else:
        with open(args.ba_json, "r", encoding="utf-8") as f:
            j = json.load(f)
        R_ab = np.array(j["R"], dtype=np.float64).reshape(3,3)
        t_ab = np.array(j["t"], dtype=np.float64).reshape(3,1)

    objA = build_objp(args.rows_a, args.cols_a, args.square_a)
    objB = build_objp(args.rows_b, args.cols_b, args.square_b)

    # RTSP 오픈(저지연)
    capE = cv2.VideoCapture(args.eye_url,   cv2.CAP_FFMPEG)
    capW = cv2.VideoCapture(args.world_url, cv2.CAP_FFMPEG)
    capE.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capW.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capE.isOpened() or not capW.isOpened():
        raise SystemExit("RTSP 열기 실패(eye/world). URL/네트워크 확인")

    samples_R, samples_t = [], []
    n = 0
    auto_on = args.auto
    last_auto = 0.0
    last_cE, last_cW = None, None

    print("[INFO] SPACE=샘플추가  BACKSPACE=삭제  E=자동토글  ENTER=계산/저장  ESC/q=종료")

    try:
        while True:
            okE, fE = capE.read()
            okW, fW = capW.read()
            if not (okE and okW):
                cv2.waitKey(1); continue

            okA, cA = find_corners(fW, args.rows_a, args.cols_a)  # A=월드 보드(월드 카메라에서)
            okB, cB = find_corners(fE, args.rows_b, args.cols_b)  # B=아이 보드(아이 카메라에서)

            # 미리보기 HUD
            vE = fE.copy(); vW = fW.copy()
            if okB: cv2.drawChessboardCorners(vE, (args.cols_b, args.rows_b), cB, okB)
            if okA: cv2.drawChessboardCorners(vW, (args.cols_a, args.rows_a), cA, okA)
            for im in [(vE, "EYE (B)", okB), (vW, "WORLD (A)", okA)]:
                cv2.putText(im[0], f"{im[1]} corners: {'OK' if im[2] else 'NO'}", (10,26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if im[2] else (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(vE, f"samples: {n}/{args.max_samples}  auto:{'ON' if auto_on else 'OFF'}",
                        (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(vW, f"samples: {n}/{args.max_samples}  auto:{'ON' if auto_on else 'OFF'}",
                        (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("EYE (board B)", vE)
            cv2.imshow("WORLD (board A)", vW)

            def try_add_sample():
                nonlocal n, last_cE, last_cW
                # PnP (재투영 체크)
                ok_w, rvec_w, tvec_w = cv2.solvePnP(objA, cA, K_w, dist_w, flags=cv2.SOLVEPNP_ITERATIVE)
                ok_e, rvec_e, tvec_e = cv2.solvePnP(objB, cB, K_e, dist_e, flags=cv2.SOLVEPNP_ITERATIVE)
                if not (ok_w and ok_e):
                    print("[WARN] solvePnP 실패"); return False

                # 재투영 오차 필터링
                def reproj_err(obj, rvec, tvec, K, dist, corners):
                    proj,_ = cv2.projectPoints(obj, rvec, tvec, K, dist)
                    d = np.linalg.norm(proj.reshape(-1,2) - corners.reshape(-1,2), axis=1)
                    return float(np.median(d))
                errW = reproj_err(objA, rvec_w, tvec_w, K_w, dist_w, cA)
                errE = reproj_err(objB, rvec_e, tvec_e, K_e, dist_e, cB)
                if errW > args.reproj_thresh_px or errE > args.reproj_thresh_px:
                    print(f"[DROP] reproj too large W:{errW:.2f}px E:{errE:.2f}px"); return False

                # 보드별 카메라 기준 pose
                R_wbA = rodrigues(rvec=rvec_w); t_wbA = tvec_w.reshape(3,1)  # boardA -> worldCam
                R_ebB = rodrigues(rvec=rvec_e); t_ebB = tvec_e.reshape(3,1)  # boardB -> eyeCam

                # eye->world = (world<-A) o (A<-B) o (B<-eye)
                R_bB_e, t_bB_e = invert_RT(R_ebB, t_ebB)          # eyeCam -> boardB
                R_bA_e, t_bA_e = compose(R_ab, t_ab, R_bB_e, t_bB_e)     # boardB->boardA 합성
                R_we,   t_we   = compose(R_wbA, t_wbA, R_bA_e, t_bA_e)   # 최종 eye->world

                samples_R.append(R_we)
                samples_t.append(t_we.reshape(3))
                n += 1
                print(f"[ADD] #{n}: |t|={np.linalg.norm(t_we):.3f} m")
                last_cE = cB.reshape(-1,2).copy()
                last_cW = cA.reshape(-1,2).copy()
                return True

            # 자동 샘플링: 코너가 있고 최근 프레임 대비 충분히 '자세 변화'가 있으면 일정 간격으로
            now = time.time()
            if auto_on and okA and okB and now - last_auto > args.auto_interval and n < args.max_samples:
                move_ok = True
                if last_cE is not None and last_cW is not None:
                    dE = np.linalg.norm(cB.reshape(-1,2) - last_cE, axis=1).mean()
                    dW = np.linalg.norm(cA.reshape(-1,2) - last_cW, axis=1).mean()
                    move_ok = (dE > args.min_motion_px) or (dW > args.min_motion_px)
                if move_ok and try_add_sample():
                    last_auto = now

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k == ord('e'):
                auto_on = not auto_on
            elif k == 8:  # BACKSPACE
                if n>0:
                    samples_R.pop(); samples_t.pop(); n-=1
                    print("[INFO] last sample removed")
            elif k == 32 and okA and okB and n < args.max_samples:  # SPACE
                try_add_sample()
            elif k in (13,10) and n>0:  # ENTER → 계산/저장
                # 1) 대략 평균 구함
                R0 = mean_rotation_quat(samples_R)
                t0 = np.median(np.stack(samples_t,0), axis=0)

                # 2) 인라이어 필터(각도/변위)
                in_R, in_t = [], []
                for R, t in zip(samples_R, samples_t):
                    dR = R @ R0.T
                    ang = rot_angle_deg(dR)
                    dt  = np.linalg.norm(t - t0)
                    if ang <= args.angle_inlier_deg and dt <= args.trans_inlier_m:
                        in_R.append(R); in_t.append(t)
                if len(in_R) >= max(5, n//2):
                    Rf = mean_rotation_quat(in_R)
                    tf = np.median(np.stack(in_t,0), axis=0)
                    kept = len(in_R)
                else:
                    Rf, tf, kept = R0, t0, n

                out = {
                    "R": Rf.reshape(-1).astype(float).tolist(),
                    "t": tf.reshape(-1).astype(float).tolist(),
                    "samples_total": int(n),
                    "samples_kept": int(kept),
                    "angle_inlier_deg": float(args.angle_inlier_deg),
                    "trans_inlier_m": float(args.trans_inlier_m),
                    "B_to_A": {
                        "R": R_ab.reshape(-1).astype(float).tolist(),
                        "t": t_ab.reshape(-1).astype(float).tolist()
                    }
                }
                os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
                with open(args.out_json, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)
                print(f"[SAVE] {args.out_json}  (kept {kept}/{n})")

    finally:
        try:
            capE.release(); capW.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
