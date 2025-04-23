import argparse
import cv2
import os
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from faster_live_portrait import FasterLivePortraitPipeline

def main():
    parser = argparse.ArgumentParser(description='Faster Live Portrait - Image to Image Animation')
    parser.add_argument('--src_image', type=str, help='Path to the source image', default="/home/user/LivePortrait/assets/examples/source/s1.jpg")
    parser.add_argument('--dri_image', type=str, help='Path to the driving image', default="/home/user/LivePortrait/assets/examples/driving/d9.jpg")
    parser.add_argument('--cfg', type=str, default="configs/trt_infer.yaml", help='Path to the inference configuration file')
    parser.add_argument('--output_image', type=str, default="output_animation.png", help='Path to save the animated output image')
    parser.add_argument('--animal', action='store_true', help='Use animal model (currently not fully supported in this script)')
    # Add paste_back argument from run.py logic if needed, defaulting based on config usually
    # parser.add_argument('--paste_back', action='store_true', default=False, help='Paste back to origin image size')

    args = parser.parse_args()

    # --- Configuration Loading ---
    if not os.path.exists(args.cfg):
        print(f"Error: Config file not found at {args.cfg}")
        return
    infer_cfg = OmegaConf.load(args.cfg)
    # Optional: Override paste_back from command line if argument added
    # infer_cfg.infer_params.flag_pasteback = args.paste_back

    # --- Pipeline Initialization ---
    print("Initializing pipeline...")
    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    print("Pipeline initialized.")

    # --- Image Loading and Preprocessing ---
    print(f"Loading source image: {args.src_image}")
    src_image = cv2.imread(args.src_image)
    if src_image is None:
        print(f"Error: Failed to load source image at {args.src_image}")
        return
    
    # Convert to RGB and resize to 512x512
    # print(src_image.dtype)
    # src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB) # Keep as BGR
    src_image = cv2.resize(src_image, (512, 512))
    # src_image = np.transpose(src_image, (2, 0, 1)) # Keep as H, W, C
    # src_image = src_image.astype(np.float32)
    # Convert to tensor and move to GPU
    # src_image_tensor = torch.from_numpy(src_image).permute(2, 0, 1).float().cuda() / 255.0

    print(f"Loading driving image: {args.dri_image}")
    dri_image = cv2.imread(args.dri_image)
    if dri_image is None:
        print(f"Error: Failed to load driving image at {args.dri_image}")
        return
    
    # Convert to RGB and resize to 512x512
    # dri_image = cv2.cvtColor(dri_image, cv2.COLOR_BGR2RGB) # Keep as BGR
    dri_image = cv2.resize(dri_image, (512, 512))
    # dri_image = np.transpose(dri_image, (2, 0, 1)) # Keep as H, W, C
    # dri_image = dri_image.astype(np.float32)
    
    # Convert to tensor and move to GPU
    # dri_image_tensor = torch.from_numpy(dri_image).permute(2, 0, 1).float().cuda() / 255.0

    # --- Animation ---
    print("Starting animation...")
    start_time = time.time()
    animated_image_np = pipe.animate_image(src_image, dri_image)
    end_time = time.time()

    if animated_image_np is None:
        print("Animation failed.")
        return

    print(f"Animation successful! Time taken: {end_time - start_time:.4f} seconds")

    # --- Save Output ---
    output_dir = os.path.dirname(args.output_image)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    try:
        # Convert back to BGR for saving
        animated_image_bgr = cv2.cvtColor(animated_image_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output_image, animated_image_bgr)
        print(f"Animated image saved to: {args.output_image}")
    except Exception as e:
        print(f"Error saving output image: {e}")


if __name__ == '__main__':
    main() 