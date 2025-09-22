#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ver5_inproc_gaze_depth_fusion_webcam.py
- VDA(Depth-Anything v2-stream) in-proc
- FaceMesh stride/EMA, 교차 유효성(res_max, r_min), intrinsics 스케일(Kref), auto_flip_dir
- [NEW] --disable_gaze / --force_center / --world_only
  - disable_gaze: 시선 추적 엔진 자체 off, 항상 노란 중앙 십자가
  - force_center: 시선 추적이 켜져 있어도 무시하고 중앙으로 고정
  - world_only: Eye 카메라/FaceMesh를 아예 열지 않음(자원 절약)
"""

import os, sys, json, time, math, argparse, contextlib, importlib, importlib.util, types
from typing import Optional
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")

import cv2
try: cv2.setNumThreads(1)
except Exception: pass

import numpy as np
import mediapipe as mp
import torch

# ---- 프로젝트 유틸 ----
from scripts.io_open import BACKENDS, open_source
from scripts.utils_framegrabber import LatestFrameGrabber
try:
    from scripts.pyav_grabber import PyAvLatestGrabber
    _HAS_PYAV = True
except Exception:
    _HAS_PYAV = False

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

def ray_depth_intersection_best_of_two(O, d, depth_m, fx, fy, cx, cy,
                                       t_min=0.1, t_max=8.0, step=0.01, patch=7, z_tol=0.05):
    hit_pos = ray_depth_intersection(O,  d, depth_m, fx, fy, cx, cy,
                                     t_min=t_min, t_max=t_max, step=step, patch=patch, z_tol=z_tol)
    hit_neg = ray_depth_intersection(O, -d, depth_m, fx, fy, cx, cy,
                                     t_min=t_min, t_max=t_max, step=step, patch=patch, z_tol=z_tol)
    res_pos = hit_pos[5] if hit_pos[5] is not None else float('inf')
    res_neg = hit_neg[5] if hit_neg[5] is not None else float('inf')
    if res_pos <= res_neg:
        return hit_pos, +1
    else:
        return hit_neg, -1

def load_extrinsic(json_path):
    with open(json_path,"r",encoding="utf-8") as f: meta=json.load(f)
    R=np.array(meta["R"],np.float32).reshape(3,3); t=np.array(meta["t"],np.float32).reshape(3,1)
    return R,t

# ===== In-proc VDA loader =====
class VDAInproc:
    _MODEL_CFG = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    def __init__(self, vda_dir: str, encoder: str='vits', metric: bool=True,
                 input_size: int=256, max_res: int=640, fp32: bool=False,
                 device: str='auto'):
        self.vda_dir = os.path.abspath(vda_dir)
        self.encoder = encoder; self.metric = bool(metric)
        self.input_size = int(input_size); self.max_res = int(max_res)
        self.fp32 = bool(fp32)

        if device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cpu':
            self.device = 'cpu'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_float32_matmul_precision('high')
        if self.device == 'cuda': torch.backends.cudnn.benchmark = True

        pkg_dir = os.path.join(self.vda_dir, "video_depth_anything")
        sys.path.append(self.vda_dir)
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
        print(f"[VDAInproc] checkpoint loaded in {(time.time()-t0)*1000:.1f} ms", flush=True)

        try:
            self.model = self.model.to(self.device).eval()
        except RuntimeError as e:
            print(f"[VDAInproc] move to {self.device} failed ({e}); fallback to CPU", flush=True)
            self.device = 'cpu'; self.model = self.model.to(self.device).eval()

        self._last_depth = None; self._last_t = 0.0
        print(f"[VDAInproc] device={self.device} fp32={self.fp32} input_size={self.input_size} max_res={self.max_res}", flush=True)

    @contextlib.contextmanager
    def _autocast_ctx(self):
        if self.device == 'cuda' and not self.fp32:
            with torch.autocast(device_type='cuda', dtype=torch.float16): yield
        else:
            yield

    def _resize_for_maxres(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        if self.max_res > 0 and max(h, w) > self.max_res:
            s = self.max_res / max(h, w)
            return cv2.resize(frame_rgb, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA)
        return frame_rgb

    def step(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = self._resize_for_maxres(rgb)
        with torch.inference_mode(), self._autocast_ctx():
            depth = self.model.infer_video_depth_one(
                rgb, input_size=self.input_size, device=self.device, fp32=self.fp32
            )
        return None if depth is None else np.asarray(depth, dtype=np.float32)

    def push_and_get(self, frame_bgr: np.ndarray, min_interval: float=0.05) -> Optional[np.ndarray]:
        t = time.time()
        if t - self._last_t < max(1e-3, float(min_interval)):
            return self._last_depth
        self._last_t = t
        t0 = time.time()
        D = self.step(frame_bgr)
        if self._last_depth is None and D is not None:
            print(f"[Depth] first inference {D.shape if hasattr(D,'shape') else None} took {(time.time()-t0)*1000:.1f} ms", flush=True)
        self._last_depth = D if D is not None else self._last_depth
        return self._last_depth

# ===== FaceMesh step =====
def single_cam_dual_iris_step(frame_bgr, face_mesh, ema_L, ema_R,
                              iris_to_eyeball_ratio=2.1, flip=False, clahe=False, draw=False):
    out = {k: None for k in ["oL","dL","cL2","rL2","oR","dR","cR2","rR2","pmid","denom"]}
    if flip: frame_bgr = cv2.flip(frame_bgr, 1)
    if clahe: frame_bgr = apply_clahe_bgr(frame_bgr)
    H,W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        cv2.putText(frame_bgr, "NO FACE", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return frame_bgr, out

    lms = res.multi_face_landmarks[0].landmark

    irisL_3d = lmidx_to_xyz(lms, LEFT_IRIS_IDX, W, H)
    cL3, nL = fit_plane_svd(irisL_3d); nL = orient_normal_to_camera(nL)
    irisL_2d = np.array([[lms[i].x*W, lms[i].y*H] for i in LEFT_IRIS_IDX], np.float32)
    cxL, cyL, rL = min_enclosing_circle_2d(irisL_2d)
    R_e_L = iris_to_eyeball_ratio * rL
    oL = cL3 - nL * R_e_L
    dL = ema_L.update(nL)

    irisR_3d = lmidx_to_xyz(lms, RIGHT_IRIS_IDX, W, H)
    cR3, nR = fit_plane_svd(irisR_3d); nR = orient_normal_to_camera(nR)
    irisR_2d = np.array([[lms[i].x*W, lms[i].y*H] for i in RIGHT_IRIS_IDX], np.float32)
    cxR, cyR, rR = min_enclosing_circle_2d(irisR_2d)
    R_e_R = iris_to_eyeball_ratio * rR
    oR = cR3 - nR * R_e_R
    dR = ema_R.update(nR)

    out.update(oL=oL, dL=dL, cL2=(int(cxL),int(cyL)), rL2=rL,
               oR=oR, dR=dR, cR2=(int(cxR),int(cyR)), rR2=rR)

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

    if draw:
        cv2.circle(frame_bgr, out["cL2"], int(rL), (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, out["cL2"], 2, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, out["cR2"], int(rR), (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, out["cR2"], 2, (255,255,255), -1, cv2.LINE_AA)
        p2L = (int(out["cL2"][0] + dL[0]*120), int(out["cL2"][1] + dL[1]*120))
        p2R = (int(out["cR2"][0] + dR[0]*120), int(out["cR2"][1] + dR[1]*120))
        cv2.arrowedLine(frame_bgr, out["cL2"], p2L, (0,255,255), 2, tipLength=0.18)
        cv2.arrowedLine(frame_bgr, out["cR2"], p2R, (0,255,255), 2, tipLength=0.18)

    return frame_bgr, out

# ===== Input open =====
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

# ===== Main =====
def main():
    ap = argparse.ArgumentParser(description="Eye×World (webcam/RTSP) + VDA (in-proc)")
    ap.add_argument("--eye_cam", type=str, required=False, default="0")
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
    ap.add_argument("--input_size", type=int, default=256)
    ap.add_argument("--max_res", type=int, default=640)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--target_fps", type=float, default=18.0, help="VDA 추론 상한 FPS")

    # intrinsics(px) & extrinsic
    ap.add_argument("--fx_w", type=float, required=True)
    ap.add_argument("--fy_w", type=float, required=True)
    ap.add_argument("--cx_w", type=float, required=True)
    ap.add_argument("--cy_w", type=float, required=True)
    ap.add_argument("--extrinsic_json", type=str, required=True)

    ap.add_argument("--Kref_w", type=int, default=None, help="intrinsics 기준 width")
    ap.add_argument("--Kref_h", type=int, default=None, help="intrinsics 기준 height")

    # FaceMesh 업데이트율
    ap.add_argument("--eye_stride", type=int, default=1, help="매 N프레임마다 FaceMesh 실행 (1=매 프레임)")

    # EMA 민첩도
    ap.add_argument("--ema", type=float, default=0.5, help="EMA alpha (0.3~0.7 권장)")

    # 교차 유효성 필터
    ap.add_argument("--res_max", type=float, default=0.20, help="교차 허용 잔차 상한(미터)")
    ap.add_argument("--r_min",   type=float, default=0.20, help="광선거리 하한(미터)")
    ap.add_argument("--draw_rejected", action="store_true", help="거부된 fix도 회색 마커로 표시")

    ap.add_argument("--auto_flip_dir", action="store_true",
                    help="광선 방향(+d/-d) 둘 다 시도해서 더 좋은 쪽 선택")

    # [NEW] 확실히 끄기/중앙 강제/월드 전용
    ap.add_argument("--disable_gaze", action="store_true",
                    help="시선 추적 완전히 끔(페이스메시/교차 전부 off), 항상 노란 중앙 십자가")
    ap.add_argument("--force_center", action="store_true",
                    help="시선 추적이 켜져 있어도 결과를 무시하고 중앙 십자만 사용")
    ap.add_argument("--world_only", action="store_true",
                    help="Eye 카메라/FaceMesh를 아예 열지 않음(자원 절약)")

    # 기타
    ap.add_argument("--iris_to_eyeball_ratio", type=float, default=2.1)
    ap.add_argument("--show_fps", action="store_true")
    ap.add_argument("--clahe_eye", action="store_true")
    ap.add_argument("--z_tol", type=float, default=0.05)
    ap.add_argument("--t_max", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--no_gui", action="store_true", help="헤드리스 실행")

    args = ap.parse_args()

    print("[Args] world_cam:", args.world_cam, "eye_cam:", args.eye_cam, flush=True)
    print(f"[Args] VDA: enc={args.encoder} metric={args.metric} input={args.input_size} max_res={args.max_res} tgt_fps={args.target_fps} dev={args.device}", flush=True)
    if args.disable_gaze: print("[Args] disable_gaze = True → always center cross", flush=True)
    if args.force_center: print("[Args] force_center = True → ignore gaze, center cross", flush=True)
    if args.world_only:   print("[Args] world_only = True → eye camera not opened", flush=True)

    # --- Open cameras ---
    print("[Main] Opening cameras...", flush=True)

    # World cam (항상)
    read_world, rel_world, flip_world_rt = _open_pyav_or_fallback(
        args.world_cam, av_backend=args.av_backend_world, pixel_format=args.pixel_format_world,
        width=args.width_world, height=args.height_world, fps=args.fps_world,
        backend_fallback=args.backend, fourcc=args.fourcc_world,
        exposure=args.exposure_world, autofocus=args.autofocus_world,
        flip=args.flip_world
    )

    # Eye cam (옵션)
    if args.world_only or args.disable_gaze or args.force_center:
        read_eye = lambda: (False, None)
        def rel_eye(): pass
        flip_eye_rt = False
        print("[Main] Eye camera skipped.", flush=True)
        face_mesh = None
        ema_L = ema_R = None
    else:
        read_eye, rel_eye, flip_eye_rt = _open_pyav_or_fallback(
            args.eye_cam, av_backend=args.av_backend_eye, pixel_format=args.pixel_format_eye,
            width=args.width_eye, height=args.height_eye, fps=args.fps_eye,
            backend_fallback=args.backend, fourcc=args.fourcc_eye,
            exposure=args.exposure_eye, autofocus=args.autofocus_eye,
            flip=args.flip_eye
        )
        print("[Main] Eye camera opened. Initializing FaceMesh...", flush=True)
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.3, min_tracking_confidence=0.3
        )
        ema_L = EMA(args.ema); ema_R = EMA(args.ema)
        print("[Main] FaceMesh ready.", flush=True)

    # Extrinsics
    R_we, t_we = load_extrinsic(args.extrinsic_json)
    print(f"[Main] Extrinsic loaded. R_we shape={R_we.shape}, t_we={t_we.reshape(-1)}", flush=True)

    # VDA
    print("[Main] Initializing VDA...", flush=True)
    vda = VDAInproc(
        vda_dir=args.vda_dir, encoder=args.encoder, metric=args.metric,
        input_size=args.input_size, max_res=args.max_res, fp32=args.fp32,
        device=args.device
    )
    print("[Main] VDA ready. Warming up world camera...", flush=True)

    # Warmup
    def _warmup(get_fn, name, timeout=3.0):
        t0 = time.time(); ok = False; shape = None
        while time.time()-t0 < timeout:
            ok, frm = get_fn()
            if ok: shape = frm.shape; break
            cv2.waitKey(1)
        print(f"[Warmup] {name}: ok={ok} shape={shape}", flush=True)
        return ok, shape
    okW, shapeW = _warmup(read_world,"world_cam")

    # K 기준 해상도
    Kref_w = args.Kref_w if args.Kref_w is not None else args.width_world
    Kref_h = args.Kref_h if args.Kref_h is not None else args.height_world
    print(f"[K] reference resolution for intrinsics: {Kref_w}x{Kref_h}", flush=True)
    fx0, fy0, cx0, cy0 = args.fx_w, args.fy_w, args.cx_w, args.cy_w
    warned_k_scale = False

    # 루프 준비
    t_last = time.time()
    world_frames = 0
    min_interval = 1.0 / max(args.target_fps, 1e-3)
    eye_frame_idx = 0
    last_gaze = None
    mesh_runs = 0
    mesh_hz = 0.0

    print("[Main] Entering main loop. Press 'q' or ESC to exit.", flush=True)
    try:
        while True:
            okW, frame_world = read_world()
            if not okW:
                time.sleep(0.005); cv2.waitKey(1); continue
            if flip_world_rt: frame_world = cv2.flip(frame_world,1)

            # Eye 처리 (필요할 때만)
            if face_mesh is not None and not (args.disable_gaze or args.force_center):
                okE, frame_eye = read_eye()
                if okE and flip_eye_rt: frame_eye = cv2.flip(frame_eye,1)

                run_mesh = (eye_frame_idx % max(1,args.eye_stride) == 0)
                if run_mesh and okE:
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
                        d_eye = d_eye*s; d_eye = d_eye/(np.linalg.norm(d_eye)+1e-8)
                        last_gaze = (O_eye, d_eye)
                    mesh_runs += 1
                eye_frame_idx += 1
            else:
                last_gaze = None  # ← disable/force일 때는 항상 None로 유지

            # World depth
            depth_m = vda.push_and_get(frame_world, min_interval=min_interval)

            # Intrinsics scaling
            Hc, Wc = frame_world.shape[:2]
            sx = Wc / float(Kref_w); sy = Hc / float(Kref_h)
            fx_eff = fx0 * sx; fy_eff = fy0 * sy; cx_eff = cx0 * sx; cy_eff = cy0 * sy
            if not warned_k_scale and (abs(sx-1.0) > 1e-6 or abs(sy-1.0) > 1e-6):
                print(f"[K] scaled intrinsics for current frame {Wc}x{Hc}: "
                      f"fx={fx_eff:.1f}, fy={fy_eff:.1f}, cx={cx_eff:.1f}, cy={cy_eff:.1f} "
                      f"(sx={sx:.3f}, sy={sy:.3f})", flush=True)
                warned_k_scale = True

            # 교차 & 유효성
            hit_txt = "no depth"
            drawn = False
            if (not args.force_center) and (last_gaze is not None) and (depth_m is not None):
                O_eye, d_eye = last_gaze
                O_w = t_we.reshape(3)
                d_w = (R_we @ d_eye.reshape(3,1)).reshape(3)
                d_w = d_w/(np.linalg.norm(d_w)+1e-8)

                if args.auto_flip_dir:
                    (P_hit,u,v,Z_d,t_hit,res), sign = ray_depth_intersection_best_of_two(
                        O_w, d_w, depth_m, fx_eff, fy_eff, cx_eff, cy_eff,
                        t_min=0.1, t_max=args.t_max, step=args.step, patch=7, z_tol=args.z_tol
                    )
                else:
                    sign = +1
                    P_hit,u,v,Z_d,t_hit,res = ray_depth_intersection(
                        O_w, d_w, depth_m, fx_eff, fy_eff, cx_eff, cy_eff,
                        t_min=0.1, t_max=args.t_max, step=args.step, patch=7, z_tol=args.z_tol
                    )

                if P_hit is not None and u is not None:
                    Rm = float(np.linalg.norm(P_hit - O_w))
                    passed = (res <= args.res_max) and (Rm >= args.r_min)
                    if passed:
                        hit_txt = f"fix: u={u}, v={v} | Z={Z_d:.2f}m | R={Rm:.2f}m | res={res:.03f}m | dir={'+' if sign>0 else '-'}"
                        if not args.no_gui:
                            cv2.drawMarker(frame_world, (u,v), (0,0,255), cv2.MARKER_CROSS, 16, 2)
                            drawn = True
                    else:
                        why = []
                        if res > args.res_max: why.append(f"res>{args.res_max:.2f}")
                        if Rm  < args.r_min:   why.append(f"R<{args.r_min:.2f}")
                        hit_txt = f"fix: rejected ({' & '.join(why)}) | dir={'+' if sign>0 else '-'}"
                        if not args.no_gui and args.draw_rejected:
                            cv2.drawMarker(frame_world, (u,v), (180,180,180), cv2.MARKER_TILTED_CROSS, 14, 1)
                            drawn = True

            # Fallback center (강제 or gaze 없음 or depth 없음)
            if (args.force_center or last_gaze is None or depth_m is None) and not drawn:
                u0, v0 = int(Wc*0.5), int(Hc*0.5)
                hit_txt = "fix: fallback=center"
                if not args.no_gui:
                    cv2.drawMarker(frame_world, (u0,v0), (0,255,255), cv2.MARKER_CROSS, 16, 2)

            # 1초마다 로그
            now = time.time()
            if now - t_last >= 1.0:
                span = now - t_last
                world_fps = world_frames / span if span > 0 else 0.0
                print(f"[FPS] world_loop={world_fps:4.1f} | {hit_txt}", flush=True)
                world_frames = 0; t_last = now

            # GUI
            if not args.no_gui:
                hud = f"VDA[{args.encoder}{'/metric' if args.metric else ''}] size={args.input_size},max={args.max_res}"
                cv2.putText(frame_world, hud, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame_world, hit_txt, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow("World (VDA in-proc + Fixation)", frame_world)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q')):
                    print("[Main] exit key pressed", flush=True)
                    break

            world_frames += 1

    finally:
        print("[Main] cleaning up...", flush=True)
        try:
            if 'face_mesh' in locals() and face_mesh is not None: face_mesh.close()
        except Exception as e: print("[Main] face_mesh.close err:", e, flush=True)
        try: rel_world()
        except Exception as e: print("[Main] rel_world err:", e, flush=True)
        try:
            if 'rel_eye' in locals(): rel_eye()
        except Exception as e: print("[Main] rel_eye err:", e, flush=True)
        try: cv2.destroyAllWindows()
        except Exception as e: print("[Main] destroyAllWindows err:", e, flush=True)
        print("[Main] bye.", flush=True)

if __name__ == "__main__":
    main()
