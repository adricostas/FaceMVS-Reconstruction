import pycolmap
import os
import subprocess
from pathlib import Path
import open3d as o3d
import numpy as np
import shutil

class ColmapEngine:
    """
    Class to handle the 3D sparse reconstruction pipeline 
    using PyCOLMAP.
    """
    def __init__(self, img_dir, output_dir):
        self.img_dir = Path(img_dir)
        self.output_dir = Path(output_dir)
        self.database_path = self.output_dir / "database.db"
        self.dense_dir = self.output_dir / "dense"
        self.reconstruction = None
   
    def run_sfm(self):
        """
        Executes the Sparse reconstruction (SfM): Feature Extraction, Matching, and Mapping.
        """
        print("[INFO] Extracting features...")
        pycolmap.extract_features(self.database_path, self.img_dir, camera_model="SIMPLE_RADIAL")
        
        print("[INFO] Matching features (Sequential)...")
        pycolmap.match_sequential(database_path=self.database_path)

        print("[INFO] Starting incremental mapping...")
        reconstructions = pycolmap.incremental_mapping(self.database_path, self.img_dir, self.output_dir)
        
        if not reconstructions:
            print("[ERROR] SfM failed.")
            return False
        
        self.reconstruction = reconstructions[0]
        print(f"[SUCCESS] Sparse model created with {len(self.reconstruction.points3D)} points.")

        stereo_path = self.dense_dir / "stereo"
        if stereo_path.exists():
            print(f"[INFO] Cleaning cache {stereo_path}...")
            shutil.rmtree(stereo_path) 
        # 1. Undistort images
        print("[INFO] Undistorting images...")
        if not self.dense_dir.exists(): 
            self.dense_dir.mkdir(parents=True)
    
        pycolmap.undistort_images(self.dense_dir, self.output_dir / "0", self.img_dir)
        return True

    def export_to_open3d(self):
        """
        Converts PyCOLMAP data structures to Open3D format for visualization and transformation.
        """
        if not self.reconstruction:
            print("[ERROR] No reconstruction available to export.")
            return None

        points_3d = []
        colors_3d = []

        # Iterate through the 3D points in the reconstruction
        for _, point3D in self.reconstruction.points3D.items():
            points_3d.append(point3D.xyz)
            colors_3d.append(point3D.color / 255.0)

        # Create Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points_3d))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors_3d))
        
        return pcd