import argparse
import shutil
from pathlib import Path
from src import FrameExtractor

def run_extraction(input_path, frames_dir):
    input_path = Path(input_path)
    output_dir = Path(frames_dir)

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        return

    # Determine if input is video or folder of images
    is_video = input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.avi', '.mov']
    is_dir = input_path.is_dir()

    if not is_video and not is_dir:
        print(f"[ERROR] Input must be a video file or a directory of images.")
        return

    # Directory Management
    if output_dir.exists():
        print(f"[INFO] Cleaning up {output_dir}...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = FrameExtractor(max_frames=1000, motion_threshold=0.0)
    
    if is_video:
        print(f"[INFO] Processing video: {input_path.name}")
        frames = extractor.load_and_filter(input_path, is_video=True)
    else:
        print(f"[INFO] Processing image directory: {input_path.name}")
        # Sort files to ensure sequential processing
        img_list = sorted([f for f in input_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        frames = extractor.load_and_filter(img_list, is_video=False)

    if not frames:
        print("[WARNING] No frames extracted.")
        return

    print(f"[INFO] Saving {len(frames)} keyframes to {output_dir}...")
    extractor.save_for_colmap(str(output_dir))
    print(f"[SUCCESS] Stage 1 Complete.")

# En el __main__, ahora basta con un argumento '--input'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to video or images folder")
    parser.add_argument("--out", type=str, default="data/keyframes")
    args = parser.parse_args()
    run_extraction(args.input, args.out)