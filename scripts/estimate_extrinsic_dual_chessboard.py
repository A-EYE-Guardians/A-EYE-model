#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
보드A(월드)·보드B(아이) 백투백 지그로 eye->world 추정 (PyAV 저지연 캡처 우선).
- eye/world 입력: PyAV(FFmpeg) 우선, 숫자 인덱스면 OpenCV 폴백
- 최신 프레임 유지(큐 비우기) → 지연 최소화
- 체스보드 두 보드(A=월드, B=아이), back-to-back 또는 보드간 R,t 제공 모두 지원
- RANSAC-유사 인라이어 필터(각도/변위) 후 합성(사원수 평균/중앙값)

키보드:
  SPACE=샘플 추가, BACKSPACE=마지막 삭제, E=자동토글, ENTER=계산/저장, ESC/q=종료
"""
import os, json, time, math, argparse
import numpy as np, cv2
from typing import List, Tuple

# ---- 폴백(OpenCV) 경로에서 사용 ----
from io_open import BACKENDS, open_source
from utils_framegrabber import LatestFrameGrabber

# ---- PyAV 저지연 그랩버 ----
try:
    from pyav_grabber import PyAvLatestGrabber  # scripts/ 경로라면 import 경로만 조정
    _HAS_PYAV = True
except Exception:
    _HAS_PYAV = False


# ===================== 체스보드/수학 유틸 =====================
def find_corners(img, rows, cols):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    patt = (cols, rows)  # OpenCV는 (cols, rows)
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
            term=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,1e-3)
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term)
    return ok, corners


def build_objp(rows, cols, square_m):
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)
    objp *= float(square_m)
    return objp


def invert_RT(R, t):
    Rt = R.T
    tt = -Rt @ t
    return Rt, tt


def compose(R2,t2,R1,t1):  # (R2,t2) o (R1,t1)
    return R2 @ R1, (R2 @ t1 + t2)


def R_to_quat(R):
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr+1.0)*2
        q[3]=0.25*S; q[0]=(R[2,1]-R[1,2])/S; q[1]=(R[0,2]-R[2,0])/S; q[2]=(R[1,0]-R[0,1])/S
    else:
        i = int(np.argmax([R[0,0],R[1,1],R[2,2]]))
        if i==0:
            S = math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
            q[3]=(R[2,1]-R[1,2])/S; q[0]=0.25*S; q[1]=(R[0,1]+R[1,0])/S; q[2]=(R[0,2]+R[2,0])/S
        elif i==1:
            S = math.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2
            q[3]=(R[0,2]-R[2,0])/S; q[0]=(R[0,1]+R[1,0])/S; q[1]=0.25*S; q[2]=(R[1,2]+R[2,1])/S
        else:
            S = math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2
            q[3]=(R[1,0]-R[0,1])/S; q[0]=(R[0,2]+R[2,0])/S; q[1]=(R[1,2]+R[2,1])/S; q[2]=0.25*S
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
        if len(qs)>0 and np.dot(qs[0], q)<0: q=-q  # 부호 정렬
        qs.append(q)
    q = np.mean(np.stack(qs,0),0)
    q /= np.linalg.norm(q)
    return quat_to_R(q)


def rot_angle_deg(R):
    tr = (np.trace(R)-1)/2
    tr = min(1.0, max(-1.0, tr))
    return math.degrees(math.acos(tr))


def load_intrinsic(json_path):
    with open(json_path,"r",encoding="utf-8") as f:
        J=json.load(f)
    if "K" in J:
        K = np.array(J["K"], dtype=np.float64).reshape(3,3)
        dist = np.array(J.get("dist",[0,0,0,0,0]), dtype=np.float64).reshape(-1,1)
    else:
        K = np.array([[J["fx"],0,J["cx"]],
                      [0,J["fy"],J["cy"]],
                      [0,0,1]], dtype=np.float64)
        dist = np.array(J.get("dist",[0,0,0,0,0]), dtype=np.float64).reshape(-1,1)
    return K, dist


# ===================== 입력 오픈 (PyAV 우선) =====================
def _open_pyav_or_fallback(
    src: str,
    *,
    av_backend: str,
    pixel_format: str,
    width: int,
    height: int,
    fps: float,
    backend_fallback: str,
    fourcc: str = None,
    exposure: float = None,
    autofocus: int = None,
    flip: int = 0
):
    """
    src가 'video=장치명' 또는 URL이면 PyAV 시도, 아니면 OpenCV 폴백.
    Returns: (read_fn, release_fn, flip_runtime_bool)
    """
    use_pyav = False
    s = str(src).strip()
    if s.lower().startswith("video=") or s.lower().startswith("rtsp://") or s.lower().startswith("http"):
        use_pyav = True
    if use_pyav and not _HAS_PYAV:
        print("[WARN] PyAV를 import하지 못해 OpenCV 폴백을 사용합니다. (pip install av)")
        use_pyav = False

    if use_pyav:
        grabber = PyAvLatestGrabber(
            device_name_or_url=s,
            backend=av_backend,
            width=width, height=height,
            fps=int(fps),
            pixel_format=pixel_format
        )
        def _read():
            return grabber.read(wait_latest=True, wait_ms=200)
        def _release():
            try: grabber.release()
            except Exception: pass
        return _read, _release, bool(flip)

    # --- OpenCV 폴백 ---
    cap, flip_cv = open_source(s, backend_fallback, width, height, fps, fourcc,
                               flip=flip, exposure=exposure, autofocus=autofocus)
    grabber = LatestFrameGrabber(cap)
    def _read():
        return grabber.read()
    def _release():
        try: grabber.release()
        except Exception: pass
    return _read, _release, bool(flip_cv)


# ===================== 메인 =====================
def main():
    ap = argparse.ArgumentParser()
    # ---- Eye / World 소스 ----
    ap.add_argument("--eye_src", type=str, required=True,
                    help="PyAV: 'video=Iriun Webcam' / 'rtsp://...'; 숫자면 OpenCV 폴백(예: 2)")
    ap.add_argument("--world_src", type=str, required=True,
                    help="PyAV: 'video=USB2.0 Camera' / 'rtsp://...'; 숫자면 OpenCV 폴백(예: 1)")

    # PyAV 전용(eye/world 분리)
    ap.add_argument("--av_backend_eye", type=str, default="dshow")
    ap.add_argument("--av_backend_world", type=str, default="dshow")
    ap.add_argument("--pixel_format_eye", type=str, default="mjpeg")
    ap.add_argument("--pixel_format_world", type=str, default="mjpeg")

    # 공통 해상도/프레임
    ap.add_argument("--width_eye", type=int, default=640)
    ap.add_argument("--height_eye", type=int, default=480)
    ap.add_argument("--fps_eye", type=float, default=30)
    ap.add_argument("--width_world", type=int, default=512)
    ap.add_argument("--height_world", type=int, default=384)
    ap.add_argument("--fps_world", type=float, default=30)

    # OpenCV 폴백에서만 사용
    ap.add_argument("--backend", type=str, default="dshow", choices=list(BACKENDS.keys()))
    ap.add_argument("--fourcc_eye", type=str, default=None)
    ap.add_argument("--fourcc_world", type=str, default=None)
    ap.add_argument("--exposure_eye", type=float, default=None)
    ap.add_argument("--exposure_world", type=float, default=None)
    ap.add_argument("--autofocus_eye", type=int, default=None)
    ap.add_argument("--autofocus_world", type=int, default=None)

    # Intrinsics
    ap.add_argument("--eye_intrinsic", type=str, required=True)
    ap.add_argument("--world_intrinsic", type=str, required=True)

    # Board A(월드), Board B(아이)
    ap.add_argument("--rows_a", type=int, required=True)
    ap.add_argument("--cols_a", type=int, required=True)
    ap.add_argument("--square_a", type=float, required=True)
    ap.add_argument("--rows_b", type=int, required=True)
    ap.add_argument("--cols_b", type=int, required=True)
    ap.add_argument("--square_b", type=float, required=True)

    # 보드B -> 보드A
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--assume_back_to_back", action="store_true",
                   help="보드 앞면 평행·정반대, 앞면-앞면 간격 thickness_m 사용")
    ap.add_argument("--thickness_m", type=float, default=0.0)
    g.add_argument("--ba_json", type=str, help="보드B→보드A 변환 JSON({R:[9], t:[3]})")

    # 샘플링/필터
    ap.add_argument("--max_samples", type=int, default=40)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--auto_interval", type=float, default=0.8)
    ap.add_argument("--min_motion_px", type=float, default=15.0)
    ap.add_argument("--reproj_thresh_px", type=float, default=1.5)
    ap.add_argument("--angle_inlier_deg", type=float, default=5.0)
    ap.add_argument("--trans_inlier_m", type=float, default=0.05)

    # 출력
    ap.add_argument("--flip_eye", type=int, default=0)
    ap.add_argument("--flip_world", type=int, default=0)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    # Intrinsics 로드
    K_e, dist_e = load_intrinsic(args.eye_intrinsic)
    K_w, dist_w = load_intrinsic(args.world_intrinsic)

    # B->A 관계
    if args.assume_back_to_back:
        R_ab = np.diag([-1.0, 1.0, -1.0])               # Ry(pi)
        t_ab = np.array([0.0, 0.0, -args.thickness_m], dtype=np.float64).reshape(3,1)
    else:
        with open(args.ba_json,"r",encoding="utf-8") as f:
            j=json.load(f)
        R_ab = np.array(j["R"], dtype=np.float64).reshape(3,3)
        t_ab = np.array(j["t"], dtype=np.float64).reshape(3,1)

    # 3D 보드 포인트
    objA = build_objp(args.rows_a, args.cols_a, args.square_a)  # 월드 보드
    objB = build_objp(args.rows_b, args.cols_b, args.square_b)  # 아이 보드

    # 입력 오픈 (PyAV 우선 → OpenCV 폴백)
    read_eye, rel_eye, flip_eye = _open_pyav_or_fallback(
        args.eye_src,
        av_backend=args.av_backend_eye, pixel_format=args.pixel_format_eye,
        width=args.width_eye, height=args.height_eye, fps=args.fps_eye,
        backend_fallback=args.backend, fourcc=args.fourcc_eye,
        exposure=args.exposure_eye, autofocus=args.autofocus_eye,
        flip=args.flip_eye
    )
    read_world, rel_world, flip_world = _open_pyav_or_fallback(
        args.world_src,
        av_backend=args.av_backend_world, pixel_format=args.pixel_format_world,
        width=args.width_world, height=args.height_world, fps=args.fps_world,
        backend_fallback=args.backend, fourcc=args.fourcc_world,
        exposure=args.exposure_world, autofocus=args.autofocus_world,
        flip=args.flip_world
    )

    samples_R, samples_t = [], []
    n = 0
    auto_on = args.auto
    last_auto = 0.0
    last_cE = last_cW = None

    print("[INFO] SPACE=샘플추가  BACKSPACE=삭제  E=자동토글  ENTER=계산/저장  ESC/q=종료")

    try:
        while True:
            okE, fE = read_eye()
            okW, fW = read_world()
            if not (okE and okW):
                cv2.waitKey(1); continue
            if flip_eye:   fE = cv2.flip(fE, 1)
            if flip_world: fW = cv2.flip(fW, 1)

            okA, cA = find_corners(fW, args.rows_a, args.cols_a)  # A=월드(월드캠)
            okB, cB = find_corners(fE, args.rows_b, args.cols_b)  # B=아이(아이캠)

            vE = fE.copy(); vW = fW.copy()
            if okB: cv2.drawChessboardCorners(vE, (args.cols_b,args.rows_b), cB, okB)
            if okA: cv2.drawChessboardCorners(vW, (args.cols_a,args.rows_a), cA, okA)
            for im, tag, ok in [(vE,"EYE(B)",okB),(vW,"WORLD(A)",okA)]:
                cv2.putText(im, f"{tag}: {'OK' if ok else 'NO'}", (10,26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if ok else (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(vE, f"samples {n}/{args.max_samples}  auto:{'ON' if auto_on else 'OFF'}",(10,52),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(vW, f"samples {n}/{args.max_samples}  auto:{'ON' if auto_on else 'OFF'}",(10,52),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow("EYE (board B)", vE); cv2.imshow("WORLD (board A)", vW)

            def try_add_sample():
                nonlocal n, last_cE, last_cW
                if not (okA and okB):
                    return False

                ok_w, rvec_w, tvec_w = cv2.solvePnP(objA, cA, K_w, dist_w, flags=cv2.SOLVEPNP_ITERATIVE)
                ok_e, rvec_e, tvec_e = cv2.solvePnP(objB, cB, K_e, dist_e, flags=cv2.SOLVEPNP_ITERATIVE)
                if not (ok_w and ok_e):
                    print("[WARN] solvePnP 실패"); return False

                def reproj_err(obj, rvec, tvec, K, dist, corners):
                    proj,_=cv2.projectPoints(obj, rvec, tvec, K, dist)
                    d=np.linalg.norm(proj.reshape(-1,2)-corners.reshape(-1,2),axis=1)
                    return float(np.median(d))

                errW = reproj_err(objA, rvec_w, tvec_w, K_w, dist_w, cA)
                errE = reproj_err(objB, rvec_e, tvec_e, K_e, dist_e, cB)
                if errW>args.reproj_thresh_px or errE>args.reproj_thresh_px:
                    print(f"[DROP] reproj W:{errW:.2f}px E:{errE:.2f}px"); return False

                R_wbA = cv2.Rodrigues(rvec_w)[0]; t_wbA = tvec_w.reshape(3,1)  # boardA -> worldCam
                R_ebB = cv2.Rodrigues(rvec_e)[0]; t_ebB = tvec_e.reshape(3,1)  # boardB -> eyeCam

                # eye->world = (world<-A) o (A<-B) o (B<-eye)
                R_bB_e, t_bB_e = invert_RT(R_ebB, t_ebB)                # eye -> boardB
                R_bA_e, t_bA_e = compose(R_ab, t_ab, R_bB_e, t_bB_e)    # eye -> boardA
                R_we,   t_we   = compose(R_wbA, t_wbA, R_bA_e, t_bA_e)  # eye -> world

                samples_R.append(R_we); samples_t.append(t_we.reshape(3)); n += 1
                print(f"[ADD] #{n}: |t|={np.linalg.norm(t_we):.3f} m")
                last_cE = cB.reshape(-1,2).copy(); last_cW = cA.reshape(-1,2).copy()
                return True

            now=time.time()
            if auto_on and okA and okB and now-last_auto>args.auto_interval and n<args.max_samples:
                move_ok=True
                if last_cE is not None and last_cW is not None:
                    dE=np.linalg.norm(cB.reshape(-1,2)-last_cE,axis=1).mean()
                    dW=np.linalg.norm(cA.reshape(-1,2)-last_cW,axis=1).mean()
                    move_ok=(dE>args.min_motion_px) or (dW>args.min_motion_px)
                if move_ok and try_add_sample():
                    last_auto=now

            k=cv2.waitKey(1)&0xFF
            if k in (27, ord('q')): break
            elif k==ord('e'): auto_on=not auto_on
            elif k==8 and n>0:
                samples_R.pop(); samples_t.pop(); n-=1; print("[INFO] last sample removed")
            elif k==32 and okA and okB and n<args.max_samples:
                try_add_sample()
            elif k in (13,10) and n>0:
                # 1) 초기 평균
                R0=mean_rotation_quat(samples_R)
                t0=np.median(np.stack(samples_t,0),axis=0)

                # 2) 인라이어 필터
                in_R,in_t=[],[]
                for R,t in zip(samples_R,samples_t):
                    dR=R@R0.T
                    ang=rot_angle_deg(dR)
                    dt=np.linalg.norm(t-t0)
                    if ang<=args.angle_inlier_deg and dt<=args.trans_inlier_m:
                        in_R.append(R); in_t.append(t)

                if len(in_R)>=max(5,n//2):
                    Rf=mean_rotation_quat(in_R)
                    tf=np.median(np.stack(in_t,0),axis=0)
                    kept=len(in_R)
                else:
                    Rf, tf, kept = R0, t0, n

                out={
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
                with open(args.out_json,"w",encoding="utf-8") as f:
                    json.dump(out,f,indent=2)
                print(f"[SAVE] {args.out_json} (kept {kept}/{n})")
    finally:
        try: rel_eye()
        except Exception: pass
        try: rel_world()
        except Exception: pass
        cv2.destroyAllWindows()


if __name__=="__main__":
    main()
