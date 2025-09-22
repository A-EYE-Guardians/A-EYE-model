#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gaze_depth_fusion_webcam.py  (Low-CPU/OOM-safe rev)
- VDA 직렬 처리(큐/워커 1개) + 세그먼트 드롭 정책
- FaceMesh 디메이트(eye_stride) + EMA 유지
- Depth 스캔 최소화(scan_interval, 현재 세그먼트만)
- OpenCV/BLAS 쓰레드 제한, GUI 옵션화
"""

import os, sys, json, time, math, glob, argparse, threading, subprocess, queue
from typing import Tuple, Optional
import cv2, numpy as np, mediapipe as mp

# ### CHANGED: 과도한 스레드 억제 (필수)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# --- 기존 유틸 임포트 ---
from scripts.io_open import BACKENDS, open_source
from scripts.utils_framegrabber import LatestFrameGrabber
try:
    from scripts.pyav_grabber import PyAvLatestGrabber
    _HAS_PYAV = True
except Exception:
    _HAS_PYAV = False

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDX  = [33,133,160,159,158,157,173,246,161,163,144,145,153,154,155,33]
RIGHT_EYE_IDX = [362,263,387,386,385,384,398,466,388,390,373,374,380,381,382,362]
LEFT_IRIS_IDX  = [468,469,470,471]
RIGHT_IRIS_IDX = [473,474,475,476]

# ---------------- Utils ----------------
class EMA:
    def __init__(self, alpha: float=0.25):
        self.alpha = float(alpha); self.v = None
    def update(self, new_v: np.ndarray):
        new_v = new_v.astype(np.float32); n = np.linalg.norm(new_v)
        if n > 1e-8: new_v /= n
        if self.v is None: self.v = new_v
        else:
            self.v = self.alpha*new_v + (1.0-self.alpha)*self.v
            n2 = np.linalg.norm(self.v);  self.v /= (n2 + 1e-8)
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
    if np.dot(n, np.array([0,0,-1], np.float32)) < 0: n = -n
    return n

def min_enclosing_circle_2d(pts2d):
    (cx,cy),r = cv2.minEnclosingCircle(pts2d.astype(np.float32))
    return float(cx), float(cy), float(r)

def project_point_K(P, fx, fy, cx, cy):
    Z = P[2];  
    if Z <= 1e-6: return None
    u = int(round(fx*(P[0]/Z) + cx)); v = int(round(fy*(P[1]/Z) + cy))
    return (u, v)

def depth_at(depth_m, u, v, patch=5):
    H,W = depth_m.shape
    r0=max(0,v-patch//2); r1=min(H,v+patch//2+1)
    c0=max(0,u-patch//2); c1=min(W,u+patch//2+1)
    roi = depth_m[r0:r1, c0:c1]
    roi = roi[np.isfinite(roi)]
    return float(np.median(roi)) if roi.size else float('nan')

def ray_depth_intersection(O,d,depth_m,fx,fy,cx,cy,t_min=0.1,t_max=8.0,step=0.01,patch=5,z_tol=0.05):
    H,W = depth_m.shape[:2]
    best = (None,None,None,float('nan'),float('nan'),float('inf')); prev_res=None; t=t_min
    while t<=t_max:
        P = O + t*d
        uv = project_point_K(P,fx,fy,cx,cy)
        if uv is not None:
            u,v = uv
            if 0<=u<W and 0<=v<H:
                Z_d = depth_at(depth_m,u,v,patch)
                if math.isfinite(Z_d):
                    res = abs(P[2]-Z_d)
                    if res<best[5]: best=(P.copy(),u,v,Z_d,t,res)
                    if res<z_tol: return (P,u,v,Z_d,t,res)
                    if prev_res is not None and res>prev_res and best[0] is not None: return best
                    prev_res=res
        t += step
    return best

def load_extrinsic(json_path):
    with open(json_path,"r",encoding="utf-8") as f: meta=json.load(f)
    R=np.array(meta["R"],np.float32).reshape(3,3); t=np.array(meta["t"],np.float32).reshape(3,1)
    return R,t

# --------------- VDA Streamer (직렬/드롭) ---------------
class VDASegmentStreamer:
    """
    ### CHANGED
    - 세그먼트를 내부 큐에 넣고 워커 1개가 순차 실행 (동시성=1)
    - 큐가 가득 차면 'drop' 정책으로 최신만 유지하여 과부하 방어
    - depth 스캔도 '현재 out_dir'만 확인하고, scan_interval로 제한
    """
    def __init__(self, vda_dir, encoder="vits", metric=True,
                 input_size=360, max_res=720, fp32=False,
                 segment_secs=1, target_fps=18.0,
                 tmp_dir="./_segments", out_root="./_vda_out",
                 concurrency=1, queue_size=1, scan_interval=0.35, max_depth_age=3.0):
        self.vda_dir=vda_dir; self.encoder=encoder; self.metric=metric
        self.input_size=int(input_size); self.max_res=int(max_res); self.fp32=bool(fp32)
        self.segment_secs=int(segment_secs); self.target_fps=float(target_fps)
        self.tmp_dir=tmp_dir; self.out_root=out_root
        self.scan_interval=float(scan_interval); self.max_depth_age=float(max_depth_age)

        os.makedirs(self.tmp_dir, exist_ok=True); os.makedirs(self.out_root, exist_ok=True)
        self.writer=None; self.seg_path=None; self.out_dir=None; self.fps=None
        self.written=0; self.seg_frame_limit=None
        self._last_depth=None; self._last_depth_time=0.0; self._last_depth_mtime=0.0
        self._last_out_dir=None

        # ### CHANGED: 워커/큐
        self.task_q = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop = False
        self.workers = []
        for _ in range(max(1,int(concurrency))):
            th = threading.Thread(target=self._worker, daemon=True); th.start(); self.workers.append(th)

        # 프레임 쓰기 간격(타겟 FPS) 제한
        self._last_write_t = 0.0

    def _build_writer(self, path, w, h, fps):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(path, fourcc, fps, (w,h))
        if not vw.isOpened():
            base,_ = os.path.splitext(path); path = base+".avi"
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))
        if not vw.isOpened(): raise RuntimeError("VideoWriter open failed.")
        return vw, path

    def push(self, frame_bgr):
        # 타겟 FPS보다 빨리 들어오면 드롭
        now = time.time()
        if self.target_fps>0 and (now - self._last_write_t) < (1.0/self.target_fps):
            return
        self._last_write_t = now

        H,W = frame_bgr.shape[:2]
        if self.writer is None:
            self.fps = 30.0 if self.target_fps<=0 else float(self.target_fps)
            self.seg_frame_limit = int(round(self.fps * self.segment_secs))
            ts = time.strftime("%Y%m%d_%H%M%S")
            base = f"seg_{ts}"
            self.seg_path = os.path.join(self.tmp_dir, base + ".mp4")
            self.writer, self.seg_path = self._build_writer(self.seg_path, W, H, self.fps)
            self.out_dir = os.path.join(self.out_root, base); os.makedirs(self.out_dir, exist_ok=True)
            self._last_out_dir = self.out_dir
            self.written = 0

        self.writer.write(frame_bgr); self.written += 1
        if self.written >= self.seg_frame_limit:
            self.writer.release(); self.writer=None
            # ### CHANGED: 큐가 찼으면 가장 오래된 것을 버리고 최신만 넣기
            task = (self.seg_path, self.out_dir)
            if self.task_q.full():
                try: self.task_q.get_nowait()
                except Exception: pass
            try: self.task_q.put_nowait(task)
            except queue.Full:
                pass  # 드롭

    def _worker(self):
        py = sys.executable if sys.executable else "python"
        while not self._stop:
            try:
                seg_path, out_dir = self.task_q.get(timeout=0.1)
            except queue.Empty:
                continue
            cmd = [py, os.path.join(self.vda_dir,"run_streaming.py"),
                   "--input_video", seg_path, "--output_dir", out_dir,
                   "--encoder", self.encoder]
            if self.metric: cmd.append("--metric")
            if self.input_size: cmd += ["--input_size", str(self.input_size)]
            if self.max_res:    cmd += ["--max_res", str(self.max_res)]
            if self.fp32:       cmd.append("--fp32")
            try:
                subprocess.run(cmd, cwd=self.vda_dir)
            except Exception:
                pass
            finally:
                self.task_q.task_done()

    def stop(self):
        self._stop = True

    def _scan_latest_depth(self):
        # ### CHANGED: scan_interval로 빈도 제한 + 현재 out_dir만 검사
        now = time.time()
        if (now - self._last_depth_time) < self.scan_interval:
            return self._last_depth

        self._last_depth_time = now
        base_dir = self._last_out_dir
        if not base_dir or not os.path.isdir(base_dir):
            return self._last_depth

        # npz 우선
        npzs = sorted(glob.glob(os.path.join(base_dir, "*.npz")))
        target_path = npzs[-1] if npzs else None
        if target_path is None:
            exrs = sorted(glob.glob(os.path.join(base_dir, "*.exr")))
            target_path = exrs[-1] if exrs else None
            is_exr = True
        else:
            is_exr = False
        if not target_path:
            return self._last_depth

        try:
            mtime = os.path.getmtime(target_path)
        except Exception:
            mtime = 0.0

        # 이미 읽었던 파일이면 스킵
        if mtime <= self._last_depth_mtime:
            return self._last_depth

        try:
            if not is_exr:
                # ### CHANGED: mmap 읽기 + with 구문 (파일 핸들 누수 방지)
                with np.load(target_path, mmap_mode='r') as z:
                    D = None
                    if "depth" in z: D = z["depth"]
                    elif "arr_0" in z: D = z["arr_0"]
                    if D is not None:
                        self._last_depth = np.array(D, dtype=np.float32, copy=False)
                        self._last_depth_mtime = mtime
                        return self._last_depth
            else:
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                img = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.ndim==3: img = img[...,0]
                    self._last_depth = img.astype(np.float32)
                    self._last_depth_mtime = mtime
                    return self._last_depth
        except Exception:
            pass
        return self._last_depth

    def get_latest_depth(self): return self._scan_latest_depth()

# --------------- FaceMesh step (디메이트) ---------------
def single_cam_dual_iris_step(frame_bgr, face_mesh, ema_L, ema_R,
                              iris_to_eyeball_ratio=2.1, flip=False, clahe=False, draw=False):
    out = {k: None for k in ["oL","dL","cL2","rL2","oR","dR","cR2","rR2","pmid","denom"]}
    if flip: frame_bgr = cv2.flip(frame_bgr, 1)
    if clahe: frame_bgr = apply_clahe_bgr(frame_bgr)
    H,W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return frame_bgr, out

    lms = res.multi_face_landmarks[0].landmark
    # draw는 옵션화
    if draw:
        def draw_eye_polyline(idxs, color=(0,255,0)):
            pts = np.array([(int(lms[i].x*W), int(lms[i].y*H)) for i in idxs], np.int32)
            cv2.polylines(frame_bgr, [pts], False, color, 1, cv2.LINE_AA)
        draw_eye_polyline(LEFT_EYE_IDX); draw_eye_polyline(RIGHT_EYE_IDX)

    # L
    irisL_3d = lmidx_to_xyz(lms, LEFT_IRIS_IDX, W, H)
    cL3, nL = fit_plane_svd(irisL_3d); nL = orient_normal_to_camera(nL)
    irisL_2d = np.array([[lms[i].x*W, lms[i].y*H] for i in LEFT_IRIS_IDX], np.float32)
    cxL, cyL, rL = min_enclosing_circle_2d(irisL_2d)
    R_e_L = iris_to_eyeball_ratio * rL
    oL = cL3 - nL * R_e_L
    dL = ema_L.update(nL)
    out.update(oL=oL, dL=dL, cL2=(int(cxL),int(cyL)), rL2=rL)
    if draw:
        cv2.circle(frame_bgr, out["cL2"], int(rL), (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, out["cL2"], 2, (255,255,255), -1, cv2.LINE_AA)

    # R
    irisR_3d = lmidx_to_xyz(lms, RIGHT_IRIS_IDX, W, H)
    cR3, nR = fit_plane_svd(irisR_3d); nR = orient_normal_to_camera(nR)
    irisR_2d = np.array([[lms[i].x*W, lms[i].y*H] for i in RIGHT_IRIS_IDX], np.float32)
    cxR, cyR, rR = min_enclosing_circle_2d(irisR_2d)
    R_e_R = iris_to_eyeball_ratio * rR
    oR = cR3 - nR * R_e_R
    dR = ema_R.update(nR)
    out.update(oR=oR, dR=dR, cR2=(int(cxR),int(cyR)), rR2=rR)
    if draw:
        cv2.circle(frame_bgr, out["cR2"], int(rR), (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, out["cR2"], 2, (255,255,255), -1, cv2.LINE_AA)

    w0 = oL - oR
    a = float(np.dot(dL,dL)); b = float(np.dot(dL,dR)); c = float(np.dot(dR,dR))
    d = float(np.dot(dL,w0));  e = float(np.dot(dR,w0))
    denom = a*c - b*b
    if abs(denom) < 1e-6:
        t=0.0; s = e/c if abs(c)>1e-9 else 0.0
    else:
        t = (b*e - c*d)/denom; s = (a*e - b*d)/denom
    p1 = oL + t*dL; p2 = oR + s*dR; pmid = 0.5*(p1+p2)
    out.update(pmid=pmid, denom=float(denom))
    return frame_bgr, out

# --------------- Input open ---------------
def _open_pyav_or_fallback(src, *, av_backend, pixel_format, width, height, fps,
                           backend_fallback, fourcc=None, exposure=None, autofocus=None, flip=False):
    use_pyav = False
    s = str(src).strip().lower()
    if s.startswith("video=") or s.startswith("rtsp://") or s.startswith("http"):
        use_pyav = True
    if use_pyav and not _HAS_PYAV:
        print("[WARN] PyAV not available → OpenCV fallback"); use_pyav = False

    if use_pyav:
        grabber = PyAvLatestGrabber(device_name_or_url=src, backend=av_backend,
                                    width=width, height=height, fps=int(fps),
                                    pixel_format=pixel_format)
        def _read(): return grabber.read(wait_latest=True, wait_ms=200)
        def _release():
            try: grabber.release()
            except Exception: pass
        return _read, _release, bool(flip)

    cap, flip_cv = open_source(src, backend_fallback, width, height, fps, fourcc,
                               flip=flip, exposure=exposure, autofocus=autofocus)
    grabber = LatestFrameGrabber(cap)
    def _read(): return grabber.read()
    def _release():
        try: grabber.release()
        except Exception: pass
    return _read, _release, bool(flip_cv)

# --------------- Main ---------------
def main():
    ap = argparse.ArgumentParser(description="Eye×World (webcam/RTSP) + VDA streaming (low-CPU)")
    ap.add_argument("--eye_cam", type=str, required=True)
    ap.add_argument("--world_cam", type=str, required=True)

    ap.add_argument("--av_backend_eye", type=str, default="dshow")
    ap.add_argument("--av_backend_world", type=str, default="dshow")
    ap.add_argument("--pixel_format_eye", type=str, default="mjpeg")
    ap.add_argument("--pixel_format_world", type=str, default="mjpeg")

    ap.add_argument("--width_eye", type=int, default=640)
    ap.add_argument("--height_eye", type=int, default=480)
    ap.add_argument("--fps_eye", type=float, default=30)
    ap.add_argument("--width_world", type=int, default=512)
    ap.add_argument("--height_world", type=int, default=384)
    ap.add_argument("--fps_world", type=float, default=30)

    ap.add_argument("--backend", type=str, default="dshow", choices=list(BACKENDS.keys()))
    ap.add_argument("--fourcc_eye", type=str, default=None)
    ap.add_argument("--fourcc_world", type=str, default=None)
    ap.add_argument("--exposure_eye", type=float, default=None)
    ap.add_argument("--exposure_world", type=float, default=None)
    ap.add_argument("--autofocus_eye", type=int, default=None)
    ap.add_argument("--autofocus_world", type=int, default=None)
    ap.add_argument("--flip_eye", action="store_true")
    ap.add_argument("--flip_world", action="store_true")

    # VDA
    ap.add_argument("--vda_dir", type=str, required=True)
    ap.add_argument("--encoder", type=str, default="vits", choices=["vits","vitb","vitl"])
    ap.add_argument("--metric", action="store_true")
    ap.add_argument("--input_size", type=int, default=360)   # ### CHANGED: 더 작게
    ap.add_argument("--max_res", type=int, default=720)      # ### CHANGED: 더 작게
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--segment_secs", type=int, default=1)
    ap.add_argument("--target_fps", type=float, default=18)  # ### CHANGED: 낮춤

    # 월드 intrinsics (px)
    ap.add_argument("--fx_w", type=float, required=True)
    ap.add_argument("--fy_w", type=float, required=True)
    ap.add_argument("--cx_w", type=float, required=True)
    ap.add_argument("--cy_w", type=float, required=True)

    ap.add_argument("--extrinsic_json", type=str, required=True)

    # 기타
    ap.add_argument("--iris_to_eyeball_ratio", type=float, default=2.1)
    ap.add_argument("--ema", type=float, default=0.25)
    ap.add_argument("--show_fps", action="store_true")
    ap.add_argument("--clahe_eye", action="store_true")
    ap.add_argument("--z_tol", type=float, default=0.05)
    ap.add_argument("--t_max", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=0.01)

    # ### CHANGED: 성능 관련 신규 옵션
    ap.add_argument("--eye_stride", type=int, default=3, help="매 N프레임마다 FaceMesh 실행")
    ap.add_argument("--scan_interval", type=float, default=0.35, help="depth 스캔 최소 간격(초)")
    ap.add_argument("--max_depth_age", type=float, default=3.0, help="깊이 캐시 허용 최대 나이(초)")
    ap.add_argument("--vda_concurrency", type=int, default=1, help="VDA 동시 실행 수(권장:1)")
    ap.add_argument("--vda_queue_size", type=int, default=1, help="세그먼트 대기 큐(권장:1)")
    ap.add_argument("--no_gui", action="store_true", help="헤드리스 실행(권장)")

    args = ap.parse_args()

    read_eye, rel_eye, flip_eye_rt = _open_pyav_or_fallback(
        args.eye_cam, av_backend=args.av_backend_eye, pixel_format=args.pixel_format_eye,
        width=args.width_eye, height=args.height_eye, fps=args.fps_eye,
        backend_fallback=args.backend, fourcc=args.fourcc_eye,
        exposure=args.exposure_eye, autofocus=args.autofocus_eye,
        flip=args.flip_eye
    )
    read_world, rel_world, flip_world_rt = _open_pyav_or_fallback(
        args.world_cam, av_backend=args.av_backend_world, pixel_format=args.pixel_format_world,
        width=args.width_world, height=args.height_world, fps=args.fps_world,
        backend_fallback=args.backend, fourcc=args.fourcc_world,
        exposure=args.exposure_world, autofocus=args.autofocus_world,
        flip=args.flip_world
    )

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3, min_tracking_confidence=0.3
    )
    ema_L = EMA(args.ema); ema_R = EMA(args.ema)
    R_we, t_we = load_extrinsic(args.extrinsic_json)

    streamer = VDASegmentStreamer(
        vda_dir=args.vda_dir, encoder=args.encoder, metric=args.metric,
        input_size=args.input_size, max_res=args.max_res, fp32=args.fp32,
        segment_secs=args.segment_secs, target_fps=args.target_fps,
        tmp_dir="./_segments", out_root="./_vda_out",
        concurrency=args.vda_concurrency, queue_size=args.vda_queue_size,
        scan_interval=args.scan_interval, max_depth_age=args.max_depth_age
    )

    fps_eye=0.0; fps_w=0.0; t_last=time.time(); world_frames=0
    fx_w,fy_w,cx_w,cy_w = args.fx_w,args.fy_w,args.cx_w,args.cy_w

    eye_frame_idx = 0
    last_gaze = None   # (O_eye, d_eye)
    try:
        while True:
            okE, frame_eye = read_eye(); okW, frame_world = read_world()
            if not (okE and okW):
                cv2.waitKey(1); continue
            if flip_eye_rt: frame_eye = cv2.flip(frame_eye,1)
            if flip_world_rt: frame_world = cv2.flip(frame_world,1)

            # ---- Eye (stride 처리) ----
            t0 = time.time()
            run_mesh = (eye_frame_idx % max(1,args.eye_stride) == 0)
            if run_mesh:
                frame_eye_vis, gaze = single_cam_dual_iris_step(
                    frame_eye, face_mesh, ema_L, ema_R,
                    iris_to_eyeball_ratio=args.iris_to_eyeball_ratio,
                    flip=False, clahe=args.clahe_eye, draw=not args.no_gui
                )
                if gaze["oL"] is not None and gaze["oR"] is not None:
                    O_eye = 0.5*(gaze["oL"] + gaze["oR"])
                    d_eye = (gaze["pmid"] - O_eye) if (gaze["pmid"] is not None) else 0.5*(gaze["dL"] + gaze["dR"])
                    H_e,W_e = frame_eye.shape[:2]
                    s = np.array([1.0, float(W_e)/float(H_e), 1.0], np.float32)
                    d_eye = d_eye*s; n=np.linalg.norm(d_eye);  d_eye/= (n+1e-8)
                    last_gaze = (O_eye, d_eye)
            else:
                frame_eye_vis = frame_eye  # no draw/compute
            eye_frame_idx += 1
            t1 = time.time(); fps_eye = 0.9*fps_eye + 0.1*(1.0/max(t1-t0,1e-3))

            # ---- World: 세그먼트 공급 & 최신 깊이 ----
            streamer.push(frame_world)
            depth_m = streamer.get_latest_depth()

            hit_txt = "no depth yet"
            if last_gaze is not None and depth_m is not None:
                O_eye, d_eye = last_gaze
                O_w = t_we.reshape(3); d_w = (R_we @ d_eye.reshape(3,1)).reshape(3); d_w/= (np.linalg.norm(d_w)+1e-8)
                P_hit,u,v,Z_d,t_hit,res = ray_depth_intersection(
                    O_w,d_w,depth_m,fx_w,fy_w,cx_w,cy_w,
                    t_min=0.1, t_max=args.t_max, step=args.step, patch=7, z_tol=args.z_tol
                )
                if P_hit is not None and u is not None:
                    if not args.no_gui:
                        cv2.drawMarker(frame_world, (u,v), (0,0,255), cv2.MARKER_CROSS, 16, 2)
                    Rm = float(np.linalg.norm(P_hit - O_w))
                    hit_txt = f"fix: u={u}, v={v} | Z={Z_d:.2f}m | R={Rm:.2f}m | res={res:.03f}m"
                else:
                    hit_txt = "fix: not found (t_max/step/z_tol)"

            now=time.time()
            if now - t_last >= 1.0:
                fps_w = world_frames/(now - t_last) if (now-t_last)>0 else 0.0
                world_frames = 0; t_last = now

            if not args.no_gui:
                if args.show_fps:
                    cv2.putText(frame_eye_vis, f"Eye FPS: {fps_eye:4.1f}", (14,28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow("Eye (FaceMesh + Gaze)", frame_eye_vis)
                hud = f"VDA[{args.encoder}{'/metric' if args.metric else ''}] size={args.input_size},max={args.max_res}"
                cv2.putText(frame_world, hud, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame_world, hit_txt, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow("World (VDA-stream + Fixation)", frame_world)
            world_frames += 1

            k = cv2.waitKey(1) & 0xFF if not args.no_gui else 255
            if k in (27, ord('q')): break

    finally:
        try: streamer.stop()
        except Exception: pass
        try: rel_eye()
        except Exception: pass
        try: rel_world()
        except Exception: pass
        face_mesh.close()
        try: cv2.destroyAllWindows()
        except Exception: pass

if __name__ == "__main__":
    main()
