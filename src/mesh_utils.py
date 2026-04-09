import open3d as o3d
import numpy as np
import time
import os

def generate_final_mesh(input_ply, output_dir, max_faces=50000):
    """
    Cleans a point cloud, filters background noise, and performs 
    Poisson Surface Reconstruction to generate a manifold mesh.
    """
    start_time = time.time()
    
    # 0. Safety check: Verify input file existence
    if not os.path.exists(input_ply):
        print(f"[ERROR] Input file not found: {os.path.abspath(input_ply)}")
        return 0

    print(f"\n--- Stage 4: Starting Meshing Process ({input_ply}) ---")

    # 1. Load Point Cloud
    pcd = o3d.io.read_point_cloud(input_ply)
    if pcd.is_empty():
        print("[ERROR] Point cloud is empty. Check Stage 3 output.")
        return 0
    
    print(f"[INFO] Initial points: {len(pcd.points)}")

    # 2. Optimized Chroma Key / Background Filter
    # Removes green screen colors and absolute black artifacts
    colors = np.asarray(pcd.colors)
    
    is_green = (colors[:, 1] > 0.4) & (colors[:, 1] > colors[:, 0]) & (colors[:, 1] > colors[:, 2])
    
    # Keep points that are not green
    pcd = pcd.select_by_index(np.where(~(is_green))[0])
    print(f"[INFO] Points after background filtering: {len(pcd.points)}")

  
    o3d.io.write_point_cloud(os.path.join(output_dir, "fused_filtered.ply"), pcd) 

    # 3. Outlier Removal & Downsampling
    # Statistical filter removes "floaters" or "soup" artifacts
    print("[INFO] Aggressive Statistical Outlier Removal...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=0.5)
    
    print("[INFO] Radius Outlier Removal (Killing isolated points)...")  
    pcd, _ = pcd.remove_radius_outlier(nb_points=15, radius=0.015)

    print("[INFO] Voxel downsampling...")  
    pcd = pcd.voxel_down_sample(voxel_size=0.008) 

    o3d.io.write_point_cloud(os.path.join(output_dir, "fused_outlierrm.ply"), pcd) 

    # 4. Normal Estimation & Orientation
    print("[INFO] Estimating and orienting normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    # Orient normals toward the front (z-axis) to prevent inverted mesh faces
    pcd.orient_normals_to_align_with_direction(np.array([0., 0., 1.]))

    # 5. Poisson Surface Reconstruction (Watertight Mesh)
    print("[INFO] Performing Poisson Surface Reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, linear_fit=True)
    
    # 6. Low-Density Trimming
    # Removes "bubbles" or artifacts where the point cloud was sparse
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # 7. Mesh Decimation (Technical Constraint: < 50k faces)
    if len(mesh.triangles) > max_faces:
        print(f"[INFO] Decimating mesh from {len(mesh.triangles)} to {max_faces} triangles...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max_faces)
   
    # 8. Smoothing
    print("[INFO] Applying Taubin Smoothing (Preserving details)...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=5, lambda_filter=0.1, mu=-0.11)
   
    # 9. Final Mesh Cleanup
    print("[INFO] Running final mesh cleanup...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    # NaNs filter
    vertices = np.asarray(mesh.vertices)
    mask = np.isnan(vertices).any(axis=1)
    if np.any(mask):
        print(f"[WARNING] Found {np.count_nonzero(mask)} vertices with NaN coordinates. Removing...")
        mesh.remove_vertices_by_mask(mask)

    mesh.remove_unreferenced_vertices()
      
    # Recalculate vertex normals for correct rendering/shading
    mesh.compute_vertex_normals()

    # 10. Save Output
    output_path = os.path.join(output_dir, "final_model.ply")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save as PLY with normals and colors
    o3d.io.write_triangle_mesh(
        output_path, 
        mesh, 
        write_vertex_normals=True, 
        write_vertex_colors=True # Ensure colors are preserved
    )
    
    duration = time.time() - start_time
    print(f"[SUCCESS] Stage 4 Complete: Model saved at {output_path}")
    print(f"[INFO] Final triangle count: {len(mesh.triangles)}")
    print(f"[INFO] Meshing time: {duration:.2f}s")
    
    return duration

if __name__ == "__main__":
    # Test path - ensures compatibility with your run.py logic
    generate_final_mesh("data/workspace/dense/fused.ply", "data/workspace/dense/final_model.ply")