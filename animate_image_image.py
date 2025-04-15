import argparse
import cv2
import os
import time
import numpy as np
from omegaconf import OmegaConf
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

def main():
    parser = argparse.ArgumentParser(description='Faster Live Portrait - Image to Image Animation')
    parser.add_argument('--src_image', required=True, type=str, help='Path to the source image')
    parser.add_argument('--dri_image', required=True, type=str, help='Path to the driving image')
    parser.add_argument('--cfg', type=str, default="configs/trt_infer.yaml", help='Path to the inference configuration file (e.g., configs/trt_infer.yaml)')
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
    # Note: is_animal is passed but the np methods might need specific animal logic implemented
    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    print("Pipeline initialized.")

    # --- Image Loading ---
    print(f"Loading source image: {args.src_image}")
    src_image_np = cv2.imread(args.src_image)
    if src_image_np is None:
        print(f"Error: Failed to load source image at {args.src_image}")
        return

    print(f"Loading driving image: {args.dri_image}")
    dri_image_np = cv2.imread(args.dri_image)
    if dri_image_np is None:
        print(f"Error: Failed to load driving image at {args.dri_image}")
        return

    # --- Animation ---
    print("Starting animation...")
    start_time = time.time()
    animated_image_np = pipe.animate_image(src_image_np, dri_image_np)
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
        cv2.imwrite(args.output_image, animated_image_np)
        print(f"Animated image saved to: {args.output_image}")
    except Exception as e:
        print(f"Error saving output image: {e}")


if __name__ == '__main__':
    main() 