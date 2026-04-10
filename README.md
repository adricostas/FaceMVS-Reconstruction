# FaceMVS-Reconstruction: Hybrid Structure-from-Motion & Multi-View Stereo Pipeline

FaceMVS-Reconstruction is a 3D reconstruction pipeline designed to transform video files or image sequences of human heads into 3D meshes. The system integrates intelligent frame extraction, sparse reconstruction (SfM), neural dense reconstruction ([PatchmatchNet](https://github.com/FangjinhuaWang/PatchmatchNet)), and Poisson surface reconstruction. 

![mesh_result](result_gif.gif)
---

## 🚀 Pipeline Overview

The pipeline is organized into five specialized stages:

### Stage 0: Intelligent Extraction

Uses **rembg** for foreground isolation and a motion-threshold keyframe selection strategy to preserve strong camera baselines while avoiding redundant near-duplicate frames.

### Stage 1: Sparse Reconstruction

Leverages **pycolmap** to estimate camera intrinsics, extrinsics, and a sparse point cloud. Sequential matching was selected since the dataset is supposed to be a video or sequential images

### Stage 2: MVS Conversion

Formats the SfM output for compatibility with **PatchmatchNet**.

### Stage 3: Neural Dense Reconstruction

Utilizes **PatchmatchNet** to generate high-fidelity depth maps and a fused dense point cloud. A Neural method was selected since faces have a lot of textureless areas that classical methods cannot deal with.

### Stage 4: Surface Meshing

Implements **Poisson Surface** Reconstruction via Open3D to generate a mesh. The final model includes vertex coloring (texture) derived from the source frames.

**Note on Texture Quality**: While the pipeline produces textured meshes, color consistency on the skin may vary due to shading gradients and illumination changes during recording. 

---

## 🛠 Prerequisites

* **OS:** Ubuntu 20.04 / 22.04 (Recommended)
* **Hardware:** NVIDIA GPU with at least 8GB VRAM (required for PatchmatchNet). Tested on RTX3060
* **Conda:** Anaconda or Miniconda installed

---

## 📦 Installation

This project uses an isolated Conda environment. This ensures that the specific CUDA toolkit versions required are contained within the environment, avoiding conflicts with system-wide drivers.

### 1. Clone the Repository

```bash
git clone https://github.com/adricostas/FaceMVS-Reconstruction.git
cd FaceMVS-Reconstruction
```

### 2. Create and Activate the Environment

```bash
conda create -n facerecon_env python=3.9 -y
conda activate facerecon_env
```

### 3. Install GPU Dependencies (Conda)

We install PyTorch and the CUDA Toolkit (12.1) directly into the environment to ensure portability:

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 4. Install Project Dependencies (Pip)

```bash
pip install -r requirements.txt
```

---
## 🎥 Dataset recording Recommendations
For optimal results, follow these guidelines when recording your video:

* **Steady Subject**: Keep the head completely still. Even a small blink or smile can ruin the dense reconstruction.

* **Constant Lighting**: Use diffuse, ambient light. Avoid moving shadows or direct flashes (specular reflections).

* **Slow Motion**: Move the camera slowly and smoothly around the subject to avoid motion blur.

---

## 🖥 Usage

The entire reconstruction pipeline is fully automated via the `run.py` script. The system is designed to be "plug-and-play," requiring minimal user intervention:

* **Smart Input Detection**: The script automatically detects whether the provided `--input` is a single video file or a folder containing an image sequence.

* **End-to-End Processing**: It handles everything from pre-processing (frame extraction and background removal) to SfM, MVS (PatchmatchNet), and the final Poisson Surface Reconstruction.

Upon completion, all generated data is stored in the output directory (default: `data/results/`). The most important file is the final 3D reconstruction:

* `final_model.ply`: This is the core output of the pipeline. It contains the final 3D mesh. This file includes baked-in texture data (vertex colors) captured during the reconstruction process, allowing for a realistic representation of the subject. For the best visualization of the geometry and vertex colors, we recommend using professional tools such as MeshLab or CloudCompare.

### Basic Execution

```bash
python run.py --input path/to/your/video.mp4 [--output path/to/output_folder]
```

### Image Folder Execution

```bash
python run.py --input path/to/images_folder/ [--output path/to/output_folder]
```

---

## 📂 Output

* **Keyframes:** `data/frames/`
* **Reconstruction Workspace:** `data/workspace/`
* **Final Mesh:** `[output_folder]/final_model.ply` (default: `data/results/`)

---

## 📊 Benchmarking

The pipeline was benchmarked on an **RTX 3060 (12GB VRAM)** using a **8-second 1080p video** captured with smooth orbital motion around the subject.

### Input Statistics
- Video duration: **8 s**
- Original extracted frames: **267**
- Selected keyframes: **28**
- Capture resolution: **1920 × 1080**
- MVS inference max dimension: **960 × 540**

### Runtime Breakdown
| Stage | Time (s) |
|---|---:|
| Extraction & Filtering  | 18.94s |
| Sparse SfM (pycolmap) | 3.89s |
| MVS format conversion | 1.16s|
| Neural Dense Reconstruction (PatchmatchNet)  | 68.51s |
| Meshing & Cleaning  | 21.01s |
| **Total** | **113.51s** |

This benchmark confirms compliance with the **sub-5-minute end-to-end runtime constraint** on a consumer RTX 3060 GPU.

### Output Mesh Statistics
- Vertices: **24743**
- Faces: **49973**


---
## ⚠️ Known Limitations

Best performance is achieved when:
- the subject remains expressionless
- eyes remain open and fixed
- hair occlusion is minimized
- diffuse lighting is used
- camera motion is smooth and slow

Micro-expressions and hair strands remain the main failure modes for dense facial reconstruction.
---
## 🔮 Future Work & Scalability
While the current pipeline provides a robust baseline for 3D reconstruction, several optimizations are envisioned for production-ready environments:

### 1. Metric Scale Recovery (Biometric Scaling)
Currently, the model is generated in an arbitrary coordinate system. To achieve real-world millimeter precision, we plan to:

* IPD Normalization: Use the average human Interpupillary Distance (63mm) as a reference to rescale the point cloud.

* Iris Estimation: Integrate MediaPipe Iris to calculate absolute distance to the camera, leveraging the near-constant human iris diameter (11.7mm).

### 2. Hybrid SfM with Semantic Landmarks
To improve the robustness of the sparse reconstruction (Stage 1), we propose:

* Landmark-Guided SfM: Injecting 3D facial landmarks from MediaPipe Face Mesh into the COLMAP process. This would provide strong geometric priors, especially in videos with fast movement or challenging angles where traditional feature matching might fail.

### 3. Advanced Texture De-lighting & Blending
The current texturing suffers from lighting inconsistencies. Future iterations will include:

* Seamless Blending: Implementing a multi-band blending algorithm to smooth out transitions between frames.

* Inverse Rendering: Using a neural "de-lighting" stage to remove shadows and specular highlights, recovering the true albedo of the skin for more realistic assets.

### 4. Parametric Model Fitting (FLAME)
For applications requiring animation or rigging, the pipeline could evolve to fit a FLAME parametric model to the dense point cloud. This would transform the raw mesh into a clean, animatable topology with separated shape and expression parameters.