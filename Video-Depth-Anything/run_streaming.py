# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Apache 2.0

import argparse
import numpy as np
import os
import torch
import time
import cv2

from video_depth_anything.video_depth_stream import VideoDepthAnything
from utils.dc_utils import save_video

def main():
    parser = argparse.ArgumentParser(description='Video Depth Anything (stream-friendly)')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='-1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='-1 means original fps')
    parser.add_argument('--metric', action='store_true', help='use metric model')
    parser.add_argument('--fp32', action='store_true', help='inference in float32 (default: float16 if CUDA)')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')

    # 🔹 추가: 디바이스 선택
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='force device (default: auto)')

    args = parser.parse_args()

    # 🔹 디바이스 결정 + 튜닝
    if args.device == 'cuda':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cpu':
        DEVICE = 'cpu'
    else:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'[VDA] device: {DEVICE}  fp32={args.fp32}  input_size={args.input_size}  max_res={args.max_res}')

    torch.set_float32_matmul_precision('high')
    if DEVICE == 'cuda':
        torch.backends.cudnn.benchmark = True

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    # weights load to cpu first for safety
    state = torch.load(f'./checkpoints/{checkpoint_name}_{args.encoder}.pth', map_location='cpu')
    video_depth_anything.load_state_dict(state, strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise SystemExit(f"[ERR] cannot open input_video: {args.input_video}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if original_fps <= 1e-6:
        original_fps = 30  # fallback

    # 🔹 리사이즈 스펙 계산 (항상 width/height가 정의되도록)
    if args.max_res > 0 and max(original_height, original_width) > args.max_res:
        scale = args.max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)
    else:
        height = original_height
        width = original_width

    fps = original_fps if args.target_fps < 0 else args.target_fps
    stride = max(round(original_fps / fps), 1)

    os.makedirs(args.output_dir, exist_ok=True)
    # 🔹 심플 로그
    with open(os.path.join(args.output_dir, "vda_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"device={DEVICE} fp32={args.fp32} input_size={args.input_size} max_res={args.max_res}\n")

    depths_vis = []   # 시각화 비디오용
    frame_count = 0
    saved_count = 0
    start = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (args.max_len > 0 and frame_count >= args.max_len):
                break

            if frame_count % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if (height, width) != (original_height, original_width):
                    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)

                # 🔹 FP16 autocast (CUDA + !fp32)
                ctx_autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if (DEVICE == "cuda" and not args.fp32) else
                    torch.nullcontext()
                )
                with torch.inference_mode(), ctx_autocast:
                    depth = video_depth_anything.infer_video_depth_one(
                        rgb, input_size=args.input_size, device=DEVICE, fp32=args.fp32
                    )  # depth: HxW float32/float16 (모델 구현에 따라)

                # 🔹 npz 저장(내 메인이 스캔)
                #    파일명은 증가번호 또는 타임스탬프 모두 OK. 정렬 용이하게 6자리로 저장
                npz_path = os.path.join(args.output_dir, f"{saved_count:06d}.npz")
                # 확실히 float32로 저장(후단에서 연산 안정)
                np.savez_compressed(npz_path, depth=np.asarray(depth, dtype=np.float32))
                saved_count += 1

                # (선택) 시각화용 비디오 저장을 원하면 아래 활성화
                depths_vis.append(depth)

            frame_count += 1
            if frame_count % 50 == 0:
                print(f"frame: {frame_count}/{total_frames}  saved_npz={saved_count}")
    finally:
        cap.release()

    end = time.time()
    print(f"time: {end - start:.2f}s, saved_npz={saved_count}")

    # 🔹 시각화 비디오 출력 (옵션)
    if len(depths_vis) > 0:
        depths_vis_arr = np.stack(depths_vis, axis=0)
        video_name = os.path.basename(args.input_video)
        depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
        save_video(depths_vis_arr, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)
        print("[OK] depth vis saved ->", depth_vis_path)

if __name__ == '__main__':
    main()
