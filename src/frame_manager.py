import cv2
import os
import numpy as np
from rembg import remove, new_session


class FrameExtractor:
    def __init__(self, max_frames=200, motion_threshold=150):
        self.max_frames = max_frames
        self.motion_threshold = motion_threshold
        self.keyframes = []
        self.feature_params = dict(
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7
        )
    
        self.session = new_session("isnet-general-use") 

    def _apply_background_removal(self, frame):
        out_rgba = remove(frame, session=self.session)
        out_rgba = np.array(out_rgba)
        mask = out_rgba[:, :, 3] > 0
        
        # Crear fondo VERDE (BGR: 0, 255, 0)
        green_bg = np.zeros_like(frame)
        green_bg[:] = [0, 255, 0] # Llenamos todo de verde
        
        # Pegar el sujeto sobre el verde
        green_bg[mask] = out_rgba[:, :, :3][mask]
        
        return green_bg

    def _get_motion(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray1 = clahe.apply(gray1)
        gray2 = clahe.apply(gray2)

        p0 = cv2.goodFeaturesToTrack(gray1, **self.feature_params)
        if p0 is None: return 0

        p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None)
        if p1 is None: return 0

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) < 5: return 0

        distances = np.linalg.norm(good_new - good_old, axis=1)
        return np.mean(distances)

    def load_and_filter(self, source, is_video=True):
        """
        Processes a sequence (video or image list) and filters by motion.
        """
        self.keyframes = []
        
        if is_video:
            cap = cv2.VideoCapture(str(source))
            def get_frames():
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    yield frame
                cap.release()
            frame_gen = get_frames()
        else:
            # source is a list of image paths
            def get_frames():
                for img_path in source:
                    frame = cv2.imread(str(img_path))
                    if frame is not None: yield frame
            frame_gen = get_frames()

        # --- Core Logic ---
        try:
            frame_prev = next(frame_gen)
        except StopIteration:
            return []

        # Save first frame
        self.keyframes.append(self._apply_background_removal(frame_prev))
        last_frame_raw = frame_prev

        for frame_curr in frame_gen:
            if len(self.keyframes) >= self.max_frames:
                break

            # Calculate motion on RAW frames
            motion = self._get_motion(last_frame_raw, frame_curr)

            if motion > self.motion_threshold:
                clean_frame = self._apply_background_removal(frame_curr)
                self.keyframes.append(clean_frame)
                last_frame_raw = frame_curr
                print(f"[INFO] Frame {len(self.keyframes)} captured.")

        return self.keyframes
    
    def save_for_colmap(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for i, frame in enumerate(self.keyframes):
            cv2.imwrite(os.path.join(output_dir, f"frame_{i:03d}.png"), frame)