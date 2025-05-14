import argparse
import cv2
import os
import numpy as np
from omegaconf import OmegaConf
from faster_live_portrait_std import FasterLivePortraitPipeline 
import time

def main():
    parser = argparse.ArgumentParser(description='Faster Live Portrait - Image to Video Animation')
    parser.add_argument('--src_image', required=True, type=str, help='Path to the source image')
    parser.add_argument('--dri_video', required=True, type=str, help='Path to the driving video')
    parser.add_argument('--cfg', type=str, default="configs/trt_infer.yaml", help='Path to the inference configuration file (e.g., configs/trt_infer.yaml)')
    parser.add_argument('--output_video', type=str, default="output_animation.mp4", help='Path to save the animated output video')

    args = parser.parse_args()

    if not os.path.exists(args.cfg):
        print(f"Error: Config file not found at {args.cfg}")
        return

    if not os.path.exists(args.src_image):
        print(f"Error: Source image not found at {args.src_image}")
        return

    if not os.path.exists(args.dri_video):
        print(f"Error: Driving video not found at {args.dri_video}")
        return

    infer_cfg = OmegaConf.load(args.cfg)
    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=False)

    ret = pipe.prepare_source(args.src_image)
    if not ret:
        print(f"Error: No face detected in source image {args.src_image} or other preparation error.")
        return
    
    src_imgs = pipe.src_imgs[0]
    src_infos = pipe.src_infos[0]  

    vcap = cv2.VideoCapture(args.dri_video)
    if not vcap.isOpened():
        print(f"Error: Cannot open driving video {args.dri_video}")
        return

    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    h, w = src_imgs.shape[:2]      

    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter(args.output_video, fourcc, fps, (w, h))

    frame_ind = 0
    total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames from {args.dri_video}...")

    start_time_total = time.time()

    while vcap.isOpened():
        ret, driving_frame_bgr = vcap.read()
        if not ret:
            break

        first_frame_flag = frame_ind == 0
        
        _, _, out_animated_frame, _ = pipe.run(driving_frame_bgr, src_imgs, src_infos, first_frame=first_frame_flag)

        if out_animated_frame is None:
            print(f"Warning: Animation failed for frame {frame_ind}. Skipping.")
            frame_ind += 1
            continue

        out_animated_frame_bgr = cv2.cvtColor(out_animated_frame, cv2.COLOR_RGB2BGR)
        vout.write(out_animated_frame_bgr)
        
        frame_ind += 1


    vcap.release()
    vout.release()
    
    end_time_total = time.time()
    print(f"Animation complete. Output video saved to: {args.output_video}")
    print(f"Total processing time: {end_time_total - start_time_total:.2f} seconds for {frame_ind} frames.")
    if frame_ind > 0 :
        print(f"Average time per frame: {(end_time_total - start_time_total)/frame_ind * 1000:.2f} ms")

if __name__ == '__main__':
    main() 