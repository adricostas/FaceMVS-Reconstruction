import os
import time
import subprocess
import argparse
import shutil
from pathlib import Path
import open3d as o3d

# Imports internos
from src import FrameExtractor, ColmapEngine
from src.mesh_utils import generate_final_mesh

def main():
    parser = argparse.ArgumentParser(description="End-to-End 3D Reconstruction Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input video or image folder")
    parser.add_argument("--output", type=str, required=False, default= "data/results", help="Path to output folder (default: data/results/)")
    args = parser.parse_args()

    # --- Configuration ---
    input_source = Path(args.input)
    output_path = Path(args.output)
    frames_dir = "data/frames"
    workspace_dir = "data/workspace"    
    checkpoint_path = "checkpoints/params_000007.ckpt"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    timings = {}
    total_start = time.perf_counter()

    # ---------------------------------------------------------
    # STAGE 0: Intelligent Frame Extraction
    # ---------------------------------------------------------
    print(f"\n[STAGE 0] Extracting & filtering keyframes from: {input_source.name}")
    extract_start = time.perf_counter()
    
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)
    if not os.path.exists(output_path):
       os.makedirs(output_path, exist_ok=True)

    #Extract keyframes
    extractor = FrameExtractor(max_frames=100, motion_threshold=50.0)
    
    is_video = input_source.is_file() and input_source.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
    
    if is_video:
        frames = extractor.load_and_filter(str(input_source), is_video=True)
    else:
        img_list = sorted([f for f in input_source.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        frames = extractor.load_and_filter(img_list, is_video=False)

    if not frames:
        print("[ERROR] No frames extracted. Pipeline stopped.")
        return
        
    extractor.save_for_colmap(frames_dir)
    timings['Extraction & Filtering'] = time.perf_counter() - extract_start

    # ---------------------------------------------------------
    # STAGE 1: Sparse Reconstruction (SfM)
    # ---------------------------------------------------------
    print("\n[STAGE 1] Starting SfM (Sparse) reconstruction...")
    sfm_start = time.perf_counter()
    
    engine = ColmapEngine(img_dir=frames_dir, output_dir=workspace_dir)
    sfm_success = engine.run_sfm()
    
    if not sfm_success:
        print("[ERROR] The SfM reconstruction failed.")
        return
    
    # Export original sparse cloud for reference
    #pcd = engine.export_to_open3d()   
    #o3d.io.write_point_cloud(os.path.join(output_path, "sparse_reference.ply"), pcd) 
    
    timings['SfM Sparse'] = time.perf_counter() - sfm_start

    # ---------------------------------------------------------
    # STAGE 2: Data Conversion (COLMAP -> MVS)
    # ---------------------------------------------------------
    print("\n[STAGE 2] Converting COLMAP data to MVS format...")
    conv_start = time.perf_counter()
    
    colmap_dense_dir = os.path.join(workspace_dir, "dense") 
    if not os.path.exists(colmap_dense_dir):
        print(f"[ERROR] Dense folder not found at {colmap_dense_dir}.")
        return

    conv_cmd = [
        "python", "patchmatchnet/colmap_input.py",
        "--input_folder", colmap_dense_dir
    ]
    
    try:
        subprocess.run(conv_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] colmap_input.py failed: {e}")
        return
        
    timings['Data Conversion'] = time.perf_counter() - conv_start

    # ---------------------------------------------------------
    # STAGE 3: Dense Reconstruction (PatchMatchNet)
    # ---------------------------------------------------------
    print("\n[STAGE 3] Starting PatchMatchNet Inference...")
    dense_start = time.perf_counter()
    
    pm_cmd = [
        "python", "patchmatchnet/eval.py",
        "--input_folder", colmap_dense_dir,
        "--output_folder", output_path,
        "--checkpoint_path", checkpoint_path,
        "--num_views", "6",
        "--image_max_dim", "960",
        "--photo_thres", "0.3",
        "--geo_mask_thres", "3"
    ]
    
    try:
        subprocess.run(pm_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] PatchMatchNet Inference failed: {e}")
        return
        
    timings['Dense Reconstruction'] = time.perf_counter() - dense_start

    # ---------------------------------------------------------
    # STAGE 4: Meshing & Cleaning (Poisson)
    # ---------------------------------------------------------
    print("\n[STAGE 4] Generating final triangle mesh...")
    mesh_start = time.perf_counter()
    
    input_ply = os.path.join(output_path, "fused.ply")   
    
    generate_final_mesh(input_ply, output_path, max_faces=50000)
    timings['Meshing & Cleaning'] = time.perf_counter() - mesh_start

    # --- FINAL REPORT ---
    total_time = time.perf_counter() - total_start
    print("\n" + "="*45)
    print(f"{'PIPELINE STAGE':<25} | {'DURATION':<12}")
    print("-" * 45)
    for stage, duration in timings.items():
        print(f"{stage:<25} | {duration:>10.2f}s")
    print("-" * 45)
    print(f"{'TOTAL WALL-CLOCK':<25} | {total_time:>10.2f}s")
    print("="*45)

if __name__ == "__main__":
    main()