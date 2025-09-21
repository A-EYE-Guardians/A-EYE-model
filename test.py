# test_vda_import.py
import os, sys, torch

print("torch", torch.__version__, "cuda_avail:", torch.cuda.is_available(), flush=True)

vda_dir = r"C:\Gukbi\Direct_RP_CV\Video-Depth-Anything"
ckpt = os.path.join(vda_dir, "checkpoints", "metric_video_depth_anything_vits.pth")
print("ckpt exists:", os.path.isfile(ckpt), "size:", (os.path.getsize(ckpt) if os.path.isfile(ckpt) else 0), flush=True)

sys.path.append(vda_dir)
from video_depth_anything.video_depth_stream import VideoDepthAnything
print("imported VideoDepthAnything OK", flush=True)

m = VideoDepthAnything(encoder="vits", features=64, out_channels=[48,96,192,384])
print("model built", flush=True)

state = torch.load(ckpt, map_location="cpu")
print("ckpt loaded to RAM", flush=True)

m.load_state_dict(state, strict=True)
print("state_dict loaded", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = m.to(dev).eval()
print("model moved to", dev, flush=True)
