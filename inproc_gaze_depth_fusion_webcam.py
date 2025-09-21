#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inproc_gaze_depth_fusion_webcam.py
- Video-Depth-Anything(VDA) 모델을 같은 프로세스에서 1회 로드하여 매 프레임 추론 (mp4 세그먼트/서브프로세스 제거)
- 저지연/저CPU/OOM-safe 지향
- FaceMesh는 eye_stride로 디메이트, EMA 유지
- 풍부한 디버그 로그 포함 (장치/캠/모델/체크포인트/워밍업/추론 첫 프레임 시간 등)
"""

import os, sys, json, time, math, argparse, contextlib, importlib, importlib.util, types
from typing import Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 스레드 과다 억제 (Windows/BLAS/OpenCV)
# ─────────────────────────────────────────────────────────────────────────────
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
import torch

# --- 프로젝트 유틸 ---
from scripts.io_open import BACKENDS, open_source
from scripts.utils_framegrabber import LatestFrameGrabber
try:
    from scripts.pyav_grabber import PyAvLatestGrabber
    _HAS_PYAV = True
except Exception:
    _HAS_PYAV = False

# ─────────────────────────────────────────────────────────────────────────────
# FaceMesh / landmark indices
# ─────────────────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDX  = [33,133,160,159,158,157,173,246,161,163,144,145,153,154,155,33]
RIGHT_EYE_IDX = [362,263,387,386,385,384,398,466,388,390,373,374,380,381,382,362]
LEFT_IRIS_IDX  = [468,469,470,471]
RIGHT_IRIS_IDX = [473,474,475,476]

# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, alpha: float=0.25):
        self.alpha = float(alpha); self.v = None
    def update(self, new_v: np.ndarray):
        new_v = new_v.astype(np.float32)
        n = np.linalg.norm(new_v)
        if n > 1e-8: new_v /= n
        if self.v is None: self.v = new_v
        else:
            self.v = self.alpha*new_v + (1.0-self.alpha)*self.v
            n2 = np.linalg.norm(self.v)
            if n2 > 1e-8: self.v /= n2
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
    Z = P[2]
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

# ─────────────────────────────────────────────────────────────────────────────
# In-Proc VDA Runner (패키지 컨텍스트 유지: relative import(.dinov2 등) 대응)
# ─────────────────────────────────────────────────────────────────────────────
class VDAInproc:
    _MODEL_CFG = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    def __init__(self, vda_dir: str, encoder: str='vits', metric: bool=True,
                 input_size: int=360, max_res: int=720, fp32: bool=False,
                 device: str='auto'):
        self.vda_dir = os.path.abspath(vda_dir)
        self.encoder = encoder
        self.metric = bool(metric)
        self.input_size = int(input_size)
        self.max_res = int(max_res)
        self.fp32 = bool(fp32)

        # 디바이스 결정
        if device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cpu':
            self.device = 'cpu'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        torch.set_float32_matmul_precision('high')
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True

        # ── VDA import (정상 패키지 임포트 우선, 실패 시 패키지 컨텍스트 수동 구성)
        pkg_dir = os.path.join(self.vda_dir, "video_depth_anything")
        sys.path.append(self.vda_dir)  # 패키지 루트를 sys.path에 추가
        try:
            vda_stream = importlib.import_module("video_depth_anything.video_depth_stream")
        except Exception:
            mod_path = os.path.join(pkg_dir, "video_depth_stream.py")
            if not os.path.isfile(mod_path):
                raise FileNotFoundError(f"[VDAInproc] missing module: {mod_path}")
            if "video_depth_anything" not in sys.modules:
                pkg = types.ModuleType("video_depth_anything")
                pkg.__path__ = [pkg_dir]
                sys.modules["video_depth_anything"] = pkg
            spec = importlib.util.spec_from_file_location(
                "video_depth_anything.video_depth_stream", mod_path
            )
            if not spec or not spec.loader:
                raise RuntimeError("[VDAInproc] failed to build module spec for VDA")
            module = importlib.util.module_from_spec(spec)
            sys.modules["video_depth_anything.video_depth_stream"] = module
            spec.loader.exec_module(module)
            vda_stream = module

        VideoDepthAnything = getattr(vda_stream, "VideoDepthAnything")

        # ── 모델 생성 + 체크포인트 로드
        cfg = self._MODEL_CFG[self.encoder]
        self.model = VideoDepthAnything(**cfg)

        ckpt_name = ('metric_video_depth_anything' if self.metric else 'video_depth_anything') + f'_{self.encoder}.pth'
        ckpt_path = os.path.join(self.vda_dir, 'checkpoints', ckpt_name)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[VDAInproc] checkpoint not found: {ckpt_path}")
        print(f"[VDAInproc] loading checkpoint: {ckpt_path}", flush=True)
        t0 = time.time()
        state = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(state, strict=True)
        t1 = time.time()
        print(f"[VDAInproc] checkpoint loaded in {(t1-t0)*1000:.1f} ms", flush=True)

        # ── 디바이스 이동 (OOM 시 CPU로 폴백)
        try:
            self.model = self.model.to(self.device).eval()
        except RuntimeError as e:
            print(f"[VDAInproc] move to {self.device} failed ({e}); falling back to CPU", flush=True)
            self.device = 'cpu'
            self.model = self.model.to(self.device).eval()

        self._last_depth = None
        self._last_t = 0.0
        print(f"[VDAInproc] device={self.device} fp32={self.fp32} input_size={self.input_size} max_res={self.max_res}", flush=True)

    @contextlib.contextmanager
    def _autocast_ctx(self):
        if self.device == 'cuda' and not self.fp32:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                yield
        else:
            yield

    def _resize_for_maxres(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        if self.max_res > 0 and max(h, w) > self.max_res:
            scale = self.max_res / max(h, w)
            neww, newh = int(round(w*scale)), int(round(h*scale))
            return cv2.resize(frame_rgb, (neww, newh), interpolation=cv2.INTER_AREA)
        return frame_rgb

    def step(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = self._resize_for_maxres(rgb)
        with torch.inference_mode(), self._autocast_ctx():
            depth = self.model.infer_video_depth_one(
                rgb, input_size=self.input_size, device=self.device, fp32=self.fp32
            )
        if depth is None:
            return None
        return np.asarray(depth, dtype=np.float32)

    def push_and_get(self, frame_bgr: np.ndarray, min_interval: float=0.05) -> Optional[np.ndarray]:
        t = time.time()
        if t - self._last_t < max(1e-3, float(min_interval)):
            return self._last_depth
        self._last_t = t
        t0 = time.time()
        D = self.step(frame_bgr)
        dt = (time.time()-t0)*1000.0
        if self._last_depth is None and D is not None:
            print(f"[Depth] first inference {D.shape if hasattr(D,'shape') else None} took {dt:.1f} ms", flush=True)
        elif D is not None and int(t) % 10 == 0:
            print(f"[Depth] infer {D.shape} {dt:.1f} ms", flush=True)
        if D is not None:
            self._last_depth = D
        return self._last_depth

# ─────────────────────────────────────────────────────────────────────────────
# FaceMesh step (stride)
# ─────────────────────────────────────────────────────────────────────────────
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
    if draw:
        def draw_eye_polyline(idxs, color=(0,255,0)):
            pts = np.array([(int(lms[i].x*W), int(lms[i].y*H)) for i in idxs], np.int32)
            cv2.polylines(frame_bgr, [pts], False, color, 1, cv2.LINE_AA)
        draw_eye_polyline(LEFT_EYE_IDX); draw_eye_polyline(RIGHT_EYE_IDX)

    # Left
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

    # Right
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

    # Cyclopean mid
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

# ─────────────────────────────────────────────────────────────────────────────
# Input open (PyAV 우선, 숫자는 OpenCV 폴백)
# ─────────────────────────────────────────────────────────────────────────────
def _open_pyav_or_fallback(src, *, av_backend, pixel_format, width, height, fps,
                           backend_fallback, fourcc=None, exposure=None, autofocus=None, flip=False):
    use_pyav = False
    s = str(src).strip().lower()
    if s.startswith("video=") or s.startswith("rtsp://") or s.startswith("http"):
        use_pyav = True
    if use_pyav and not _HAS_PYAV:
        print("[WARN] PyAV not available → OpenCV fallback", flush=True)
        use_pyav = False

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

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Eye×World (webcam/RTSP) + VDA (in-proc, low-CPU)")
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

    # VDA(in-proc)
    ap.add_argument("--vda_dir", type=str, required=True)
    ap.add_argument("--encoder", type=str, default="vits", choices=["vits","vitb","vitl"])
    ap.add_argument("--metric", action="store_true")
    ap.add_argument("--input_size", type=int, default=360)
    ap.add_argument("--max_res", type=int, default=720)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--target_fps", type=float, default=18.0, help="VDA 추론 상한 FPS")

    # 월드 intrinsics (px)
    ap.add_argument("--fx_w", type=float, required=True)
    ap.add_argument("--fy_w", type=float, required=True)
    ap.add_argument("--cx_w", type=float, required=True)
    ap.add_argument("--cy_w", type=float, required=True)

    # 외부파라미터: eye → worldCam (meters)
    ap.add_argument("--extrinsic_json", type=str, required=True)

    # 기타
    ap.add_argument("--iris_to_eyeball_ratio", type=float, default=2.1)
    ap.add_argument("--ema", type=float, default=0.25)
    ap.add_argument("--show_fps", action="store_true")
    ap.add_argument("--clahe_eye", action="store_true")
    ap.add_argument("--z_tol", type=float, default=0.05)
    ap.add_argument("--t_max", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--eye_stride", type=int, default=3, help="매 N프레임마다 FaceMesh 실행")
    ap.add_argument("--no_gui", action="store_true", help="헤드리스 실행")

    args = ap.parse_args()

    # ── 인자 요약 로그
    print("[Args] eye_cam:", args.eye_cam, "world_cam:", args.world_cam, flush=True)
    print(f"[Args] VDA: encoder={args.encoder} metric={args.metric} input_size={args.input_size} max_res={args.max_res} target_fps={args.target_fps} device={args.device} fp32={args.fp32}", flush=True)
    print(f"[Args] Intrinsics fx={args.fx_w} fy={args.fy_w} cx={args.cx_w} cy={args.cy_w}", flush=True)
    if any(abs(v) < 1e-6 for v in [args.fx_w, args.fy_w]):
        print("[WARN] fx/fy가 0 또는 비정상입니다. intrinsics 전달을 다시 확인하세요.", flush=True)

    # ── 카메라 오픈
    print("[Main] Opening cameras...", flush=True)
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
    print("[Main] Cameras opened. Initializing FaceMesh...", flush=True)

    # ── FaceMesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.3, min_tracking_confidence=0.3
    )
    ema_L = EMA(args.ema); ema_R = EMA(args.ema)
    print("[Main] FaceMesh ready. Loading extrinsics...", flush=True)

    # ── Extrinsic
    R_we, t_we = load_extrinsic(args.extrinsic_json)
    print(f"[Main] Extrinsic loaded. R_we shape={R_we.shape}, t_we={t_we.reshape(-1)}", flush=True)

    # ── VDA 초기화
    print("[Main] Initializing VDA...", flush=True)
    vda = VDAInproc(
        vda_dir=args.vda_dir, encoder=args.encoder, metric=args.metric,
        input_size=args.input_size, max_res=args.max_res, fp32=args.fp32,
        device=args.device
    )
    print("[Main] VDA ready. Warming up cameras...", flush=True)

    # ── 카메라 워밍업(프레임 공급 확인)
    def _warmup(get_fn, name, timeout=3.0):
        t0 = time.time()
        ok = False; shape = None
        while time.time() - t0 < timeout:
            ok, frm = get_fn()
            if ok:
                shape = frm.shape
                break
            cv2.waitKey(1)
        print(f"[Warmup] {name}: ok={ok} shape={shape}", flush=True)
        return ok

    okE = _warmup(read_eye, "eye_cam")
    okW = _warmup(read_world, "world_cam")
    if not (okE and okW):
        print("[Warmup] Camera not delivering frames. Check indices/resolution/backend.", flush=True)
        # 종료 전 리소스 클린업
        try: rel_eye()
        except: pass
        try: rel_world()
        except: pass
        try: face_mesh.close()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass
        return

    # ── 루프 준비
    fps_eye=0.0; fps_w=0.0
    fx_w,fy_w,cx_w,cy_w = args.fx_w,args.fy_w,args.cx_w,args.cy_w
    t_last=time.time(); world_frames=0
    min_interval = 1.0/max(args.target_fps, 1e-3)

    eye_frame_idx = 0
    last_gaze = None   # (O_eye, d_eye)

    print("[Main] Entering main loop. Press 'q' or ESC to exit.", flush=True)

    try:
        while True:
            okE, frame_eye = read_eye(); okW, frame_world = read_world()
            if not (okE and okW):
                time.sleep(0.005)
                cv2.waitKey(1)
                continue

            if flip_eye_rt: frame_eye = cv2.flip(frame_eye,1)
            if flip_world_rt: frame_world = cv2.flip(frame_world,1)

            # ── Eye (stride 처리)
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
                frame_eye_vis = frame_eye
            eye_frame_idx += 1
            t1 = time.time(); fps_eye = 0.9*fps_eye + 0.1*(1.0/max(t1-t0,1e-3))

            # ── World: in-proc depth
            depth_m = vda.push_and_get(frame_world, min_interval=min_interval)

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

            # ── 1초마다 FPS/상태 로그
            now=time.time()
            if now - t_last >= 1.0:
                fps_w = world_frames/(now - t_last) if (now-t_last)>0 else 0.0
                print(f"[FPS] eye={fps_eye:4.1f} world_loop={fps_w:4.1f}  |  {hit_txt}", flush=True)
                world_frames = 0; t_last = now

            if not args.no_gui:
                if args.show_fps:
                    cv2.putText(frame_eye_vis, f"Eye FPS: {fps_eye:4.1f}", (14,28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow("Eye (FaceMesh + Gaze)", frame_eye_vis)
                hud = f"VDA[{args.encoder}{'/metric' if args.metric else ''}] size={args.input_size},max={args.max_res}"
                cv2.putText(frame_world, hud, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame_world, hit_txt, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow("World (VDA in-proc + Fixation)", frame_world)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q')):
                    print("[Main] exit key pressed", flush=True)
                    break
            # no_gui 모드에서는 Ctrl+C 로 종료

            world_frames += 1

    finally:
        print("[Main] cleaning up...", flush=True)
        try: rel_eye()
        except Exception as e: print("[Main] rel_eye err:", e, flush=True)
        try: rel_world()
        except Exception as e: print("[Main] rel_world err:", e, flush=True)
        try: face_mesh.close()
        except Exception as e: print("[Main] face_mesh.close err:", e, flush=True)
        try: cv2.destroyAllWindows()
        except Exception as e: print("[Main] destroyAllWindows err:", e, flush=True)
        print("[Main] bye.", flush=True)

if __name__ == "__main__":
    main()
