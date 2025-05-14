import argparse
import cv2
import os
import time
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from faster_live_portrait_std.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
import statistics

def main():
    parser = argparse.ArgumentParser(description='Benchmark Faster Live Portrait - Image to Image Animation')
    parser.add_argument('--src_image', required=True, type=str, help='Path to the source image')
    parser.add_argument('--dri_image', required=True, type=str, help='Path to the driving image')
    parser.add_argument('--cfg', type=str, default="configs/trt_infer.yaml", help='Path to the inference configuration file')
    parser.add_argument('--warmup_runs', type=int, default=5, help='Number of warmup runs before benchmarking')
    parser.add_argument('--benchmark_runs', type=int, default=20, help='Number of benchmark runs')
    parser.add_argument('--animal', action='store_true', help='Use animal model (currently not fully supported)')

    args = parser.parse_args()

    # --- Configuration Loading ---
    if not os.path.exists(args.cfg):
        print(f"Error: Config file not found at {args.cfg}")
        return
    infer_cfg = OmegaConf.load(args.cfg)

    # --- Pipeline Initialization ---
    print("Initializing pipeline...")
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

    # --- Warmup Phase ---
    print(f"Starting warmup ({args.warmup_runs} runs)...")
    for i in range(args.warmup_runs):
        _ = pipe.animate_image(src_image_np, dri_image_np)
        print(f"Warmup run {i+1}/{args.warmup_runs} completed.")
    print("Warmup finished.")

    # --- Benchmarking Phase ---
    print(f"Starting benchmarking ({args.benchmark_runs} runs)...")
    timings = []
    for i in tqdm(range(args.benchmark_runs)):
        start_time = time.perf_counter() # Use perf_counter for more precise timing
        animated_image_np = pipe.animate_image(src_image_np, dri_image_np)
        end_time = time.perf_counter()

        if animated_image_np is None:
            print(f"Warning: Animation failed during benchmark run {i+1}. Skipping timing.")
            continue

        duration = end_time - start_time
        timings.append(duration)

    print("Benchmarking finished.")

    # --- Results ---
    if not timings:
        print("No successful benchmark runs to analyze.")
        return

    total_time = sum(timings)
    avg_time = statistics.mean(timings)
    min_time = min(timings)
    max_time = max(timings)
    p50_time = statistics.median(timings)
    p90_time = np.percentile(timings, 90)
    p99_time = np.percentile(timings, 99)
    std_dev = statistics.stdev(timings) if len(timings) > 1 else 0

    print("\n--- Benchmark Results ---")
    print(f"Total benchmark runs: {len(timings)}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per run: {avg_time:.4f} seconds")
    print(f"Minimum time: {min_time:.4f} seconds")
    print(f"Maximum time: {max_time:.4f} seconds")
    print(f"Median (P50) time: {p50_time:.4f} seconds")
    print(f"90th Percentile time: {p90_time:.4f} seconds")
    print(f"99th Percentile time: {p99_time:.4f} seconds")
    print(f"Standard Deviation: {std_dev:.4f} seconds")
    print(f"Frames Per Second (FPS, based on average): {1.0 / avg_time:.2f}")


if __name__ == '__main__':
    main() 