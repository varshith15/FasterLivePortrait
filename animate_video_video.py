import argparse
import cv2
import os
import time
import numpy as np
from omegaconf import OmegaConf
import subprocess
import tempfile
from tqdm import tqdm  # Import tqdm for progress bar

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

def extract_audio(video_path, audio_output_path):
    """Extracts audio from video using ffmpeg."""
    command = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'copy', audio_output_path
    ]
    try:
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Audio extracted successfully to {audio_output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        print(f"FFmpeg stdout: {e.stdout.decode()}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure ffmpeg is installed and in your PATH.")
        return False

def merge_video_audio(video_path, audio_path, output_path):
    """Merges video and audio using ffmpeg."""
    command = [
        'ffmpeg', '-y', '-i', video_path, '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', output_path
    ]
    try:
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video and audio merged successfully to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error merging video and audio: {e}")
        print(f"FFmpeg stdout: {e.stdout.decode()}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure ffmpeg is installed and in your PATH.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Faster Live Portrait - Video to Video Animation')
    parser.add_argument('--src_video', required=True, type=str, help='Path to the source video')
    parser.add_argument('--dri_video', required=True, type=str, help='Path to the driving video')
    parser.add_argument('--cfg', type=str, default="configs/trt_infer.yaml", help='Path to the inference configuration file')
    parser.add_argument('--output_video', type=str, default="output_animation.mp4", help='Path to save the animated output video')
    parser.add_argument('--animal', action='store_true', help='Use animal model (compatibility depends on pipeline implementation)')
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
    try:
        # Note: is_animal is passed but the np methods might need specific animal logic implemented
        pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
        print("Pipeline initialized.")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return

    # --- Video Loading ---
    print(f"Loading source video: {args.src_video}")
    cap_src = cv2.VideoCapture(args.src_video)
    if not cap_src.isOpened():
        print(f"Error: Failed to open source video at {args.src_video}")
        return

    print(f"Loading driving video: {args.dri_video}")
    cap_dri = cv2.VideoCapture(args.dri_video)
    if not cap_dri.isOpened():
        print(f"Error: Failed to open driving video at {args.dri_video}")
        cap_src.release()
        return

    # --- Video Properties ---
    frame_width = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_src.get(cv2.CAP_PROP_FPS)
    src_frame_count = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))
    dri_frame_count = int(cap_dri.get(cv2.CAP_PROP_FRAME_COUNT))

    if src_frame_count <= 0 or dri_frame_count <= 0 :
         print(f"Warning: Could not determine frame count accurately for one or both videos.")
         # Attempt to get frame count by iterating if CAP_PROP_FRAME_COUNT fails
         # This is slower but more reliable for some formats
         if src_frame_count <= 0:
             print("Counting source frames manually...")
             src_frame_count = 0
             while True:
                 ret, _ = cap_src.read()
                 if not ret: break
                 src_frame_count += 1
             cap_src.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset position
             print(f"Source frame count (manual): {src_frame_count}")

         if dri_frame_count <= 0:
             print("Counting driving frames manually...")
             dri_frame_count = 0
             while True:
                 ret, _ = cap_dri.read()
                 if not ret: break
                 dri_frame_count += 1
             cap_dri.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset position
             print(f"Driving frame count (manual): {dri_frame_count}")

    if src_frame_count > 0 and dri_frame_count > 0 and src_frame_count != dri_frame_count:
         print(f"Warning: Source video ({src_frame_count} frames) and driving video ({dri_frame_count} frames) have different lengths.")
         print("Processing up to the shorter video length.")
         # No need to modify frame_count here, the loop will handle it

    frame_count_to_process = min(src_frame_count, dri_frame_count) if src_frame_count > 0 and dri_frame_count > 0 else max(src_frame_count, dri_frame_count) # Fallback if one count failed
    if frame_count_to_process <= 0:
        print("Error: Could not determine a valid number of frames to process.")
        cap_src.release()
        cap_dri.release()
        return

    print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")
    print(f"Processing {frame_count_to_process} frames.")


    # --- Output Setup ---
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Create temporary files for intermediate video and audio
    temp_video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".aac", delete=False).name # Use AAC for compatibility

    # Use 'mp4v' codec for MP4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(temp_video_file, fourcc, fps, (frame_width, frame_height))
    if not out_video.isOpened():
       print(f"Error: Failed to open video writer for temporary file {temp_video_file}")
       cap_src.release()
       cap_dri.release()
       # Clean up temp files if created
       if os.path.exists(temp_video_file): os.remove(temp_video_file)
       if os.path.exists(temp_audio_file): os.remove(temp_audio_file)
       return

    # --- Animation Loop ---
    print("Starting animation...")
    total_time = 0
    processed_frames = 0

    # Use tqdm for progress bar
    pbar = tqdm(total=frame_count_to_process, desc="Animating Frames")

    while processed_frames < frame_count_to_process:
        ret_src, src_frame_np = cap_src.read()
        ret_dri, dri_frame_np = cap_dri.read()

        # Break if either video ends prematurely
        if not ret_src or not ret_dri:
            print(f"Warning: Reached end of one video before expected frame count ({processed_frames}/{frame_count_to_process}).")
            break

        if src_frame_np is None or dri_frame_np is None:
             print(f"Warning: Skipped frame {processed_frames+1} due to read error.")
             processed_frames += 1
             pbar.update(1)
             continue

        start_time = time.time()
        try:
            animated_image_np = pipe.animate_image(src_frame_np, dri_frame_np)
        except Exception as e:
            print(f"Error during animation processing frame {processed_frames+1}: {e}")
            # Decide whether to continue or break. Let's try continuing.
            # Fallback: write the original source frame? Or skip? Let's skip.
            processed_frames += 1
            pbar.update(1)
            continue # Skip writing this frame

        end_time = time.time()
        total_time += (end_time - start_time)

        if animated_image_np is None:
            print(f"Warning: Animation failed for frame {processed_frames+1}. Skipping frame.")
            # Fallback: Optionally write original src_frame_np?
            # out_video.write(src_frame_np)
        else:
            # Ensure the output frame has the correct dimensions if paste_back wasn't used or failed
            if animated_image_np.shape[0] != frame_height or animated_image_np.shape[1] != frame_width:
                animated_image_np = cv2.resize(animated_image_np, (frame_width, frame_height))
            out_video.write(animated_image_np)

        processed_frames += 1
        pbar.update(1) # Update progress bar

    pbar.close() # Close progress bar

    # --- Release Resources ---
    cap_src.release()
    cap_dri.release()
    out_video.release()
    cv2.destroyAllWindows() # Good practice, though maybe not strictly needed

    if processed_frames == 0:
        print("Animation failed: No frames were processed.")
        # Clean up temp files
        if os.path.exists(temp_video_file): os.remove(temp_video_file)
        if os.path.exists(temp_audio_file): os.remove(temp_audio_file)
        return

    print(f"Animation complete for {processed_frames} frames.")
    print(f"Average processing time per frame: {total_time / processed_frames:.4f} seconds")

    # --- Audio Handling ---
    print("Extracting audio from source video...")
    if extract_audio(args.src_video, temp_audio_file):
        print("Merging animated video with source audio...")
        if not merge_video_audio(temp_video_file, temp_audio_file, args.output_video):
            print("Error during final merge. The video without audio is available at:", temp_video_file)
            # Keep the temporary video file in case of merge error, but delete audio
            if os.path.exists(temp_audio_file): os.remove(temp_audio_file)
            return # Exit after merge failure
    else:
        print("Failed to extract audio. Saving video without audio.")
        # If audio extraction fails, move the temp video to the final output path
        try:
            os.replace(temp_video_file, args.output_video) # More atomic than copy+delete
            print(f"Video (no audio) saved to: {args.output_video}")
        except OSError as e:
             print(f"Error moving temporary video file {temp_video_file} to {args.output_video}: {e}")
             print("The video without audio is available at:", temp_video_file)
             # Keep the temporary video file if move fails
             if os.path.exists(temp_audio_file): os.remove(temp_audio_file) # Still remove audio temp
             return # Exit


    # --- Cleanup ---
    print("Cleaning up temporary files...")
    if os.path.exists(temp_video_file):
        # Don't remove if it became the final output due to audio extraction failure + successful move
        if not (not os.path.exists(temp_audio_file) and os.path.abspath(temp_video_file) == os.path.abspath(args.output_video)):
             os.remove(temp_video_file)
    if os.path.exists(temp_audio_file):
        os.remove(temp_audio_file)

    print(f"Processing finished. Output video saved to: {args.output_video}")


if __name__ == '__main__':
    main() 