"""
Compile restylization results
Usage:
    python3 compile_old_lady.py --input_folder "/mnt/c/Users/koich/Siggraph/web/lady" --output_folder "./lady"

To gemini:
Can you change the code in a way that it output videos in two rows?
Source Track 
Source Video 
| Reference Track1 | Reference Track 2... | Edited Video Track |
| Reference imgae 1| Reference Image 2....| Generated Video |   
"""

import os
import sys
import argparse
import numpy as np
import cv2
import imageio
from PIL import Image

# Fix for moviepy 2.0+ breaking changes
try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    import moviepy.video.io.ImageSequenceClip as ImageSequenceClip

import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

# Local import
sys.path.append(".")
import stack_videos

def save_video_jordan(output_path, video, fps, quality=None, imageio_params=None):
    """Standard video saving function using imageio"""
    imageio_params = imageio_params if imageio_params is not None else {}
    if quality is not None:
        imageio_params["quality"] = quality
    if os.path.splitext(output_path)[1].lower() == ".gif":
        imageio_params["loop"] = 0

    writer = imageio.get_writer(output_path, fps=fps, **imageio_params)
    for i in range(len(video)):
        writer.append_data(np.array(video[i]))
    writer.close()

class Visualizer:
    def __init__(
        self,
        save_dir: str = "./results",
        pad_value: int = 0,
        fps: int = 10,
        mode: str = "rainbow",
        linewidth: int = 1,
        tracks_leave_trace: int = 0,
    ):
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
            
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps

    def visualize(self, video: np.ndarray, tracks: np.ndarray, visibility: np.ndarray = None, filename: str = "video", save_video: bool = True):
        # video is (T, H, W, C). Pad H and W.
        if self.pad_value > 0:
            pad_width = ((0,0), (self.pad_value, self.pad_value), (self.pad_value, self.pad_value), (0,0))
            video = np.pad(video, pad_width, mode='constant', constant_values=255)
            tracks = tracks + self.pad_value

        tracking_video = self.draw_tracks_on_video(video=video, tracks=tracks, visibility=visibility, filename=filename)

        if save_video:
            tracking_dir = os.path.join(self.save_dir, "tracking")
            os.makedirs(tracking_dir, exist_ok=True)
            self.save_video_clip(tracking_video, filename=filename+"_tracking", savedir=tracking_dir)
            
        return tracking_video

    def save_video_clip(self, video, filename, savedir=None):
        if savedir is None:
            save_path = os.path.join(self.save_dir, f"{filename}.mp4")
        else:
            save_path = os.path.join(savedir, f"{filename}.mp4")
            
        if isinstance(video, np.ndarray):
            video_list = list(video)
        else:
            video_list = video

        try:
            clip = ImageSequenceClip(video_list, fps=self.fps)
            clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)
        except Exception:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip as ISC
            clip = ISC(video_list, fps=self.fps)
            clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)

    def draw_tracks_on_video(self, video, tracks, visibility=None, filename=""):
        T, H, W, C = video.shape
        _, N, D = tracks.shape
        
        res_video = [frame.copy().astype(np.uint8) for frame in video]
        vector_colors = np.zeros((T, N, 3))

        if self.mode == "rainbow":
            x_min, x_max = 1e9, -1e9
            y_min, y_max = 1e9, -1e9
            
            for num_tracks in range(N):
                if visibility is not None:
                    vis_col = visibility[:, num_tracks, 0]
                    f = np.argmax(vis_col != 0)
                else:
                    f = 0
                
                x_min = min(tracks[f, num_tracks, 0], x_min)
                x_max = max(tracks[f, num_tracks, 0], x_max)
                y_min = min(tracks[f, num_tracks, 1], y_min)
                y_max = max(tracks[f, num_tracks, 1], y_max)

            safe_depth = tracks[0, :, 2].copy()
            safe_depth[safe_depth == 0] = 1.0 
            z_inv = 1.0 / safe_depth
            z_min, z_max = np.percentile(z_inv, [2, 98])
            
            norm_x = plt.Normalize(x_min, x_max)
            norm_y = plt.Normalize(y_min, y_max)
            norm_z = plt.Normalize(z_min, z_max)

            for n in range(N):
                if visibility is not None:
                    f = np.argmax(visibility[:, n, 0] != 0)
                else:
                    f = 0
                
                r = norm_x(tracks[f, n, 0])
                g = norm_y(tracks[f, n, 1])
                d_val = tracks[0, n, 2] if tracks[0, n, 2] != 0 else 1.0
                b = norm_z(1.0 / d_val)
                
                color = np.array([r, g, b])[None] * 255
                vector_colors[:, n] = np.repeat(color, T, axis=0)

        for t in tqdm(range(T), desc=f"Drawing tracks {filename}"):
            points_info = []
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                depth = tracks[t, i, 2]
                is_visible = True
                if visibility is not None:
                    is_visible = visibility[t, i, 0] > 0
                
                if coord[0] != 0 and coord[1] != 0:
                      points_info.append((i, coord, depth, is_visible))
            
            points_info.sort(key=lambda x: x[2], reverse=True)
            
            for i, coord, _, is_visible in points_info:
                if is_visible:
                    cv2.circle(res_video[t], (int(coord[0]), int(coord[1])), int(self.linewidth * 2), vector_colors[t, i].tolist(), thickness=-1)

        return np.stack(res_video)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_static_vis_video(track_npy_path, video_basename, suffix="static_ref"):
    """
    Generates a STATIC Track visualization from a specific .npy file.
    Used for Reference Tracks (where we want to visualize the ref .npy)
    """
    if not os.path.exists(track_npy_path):
        print(f"Warning: Track file not found at {track_npy_path}")
        return None

    # Unique folder for this specific visualization
    tmp_folder = os.path.join("./../tmp_webpage/tmp_vis_custom", video_basename + "_" + suffix)
    os.makedirs(tmp_folder, exist_ok=True)
    
    output_vis_path = os.path.join(tmp_folder, "tracking/overlay_tracking.mp4")
    
    if os.path.exists(output_vis_path):
        return output_vis_path

    track = np.load(track_npy_path, allow_pickle=True).item()
    
    # 1. Create black frames
    frames = np.zeros((49, track["H"], track["W"], 3), dtype=np.uint8)
    
    vis = Visualizer(save_dir=tmp_folder, linewidth=2, mode="rainbow", fps=15, tracks_leave_trace=0)

    # 2. Get tracks and FREEZE them (use first frame only)
    tr = track["uvz"]
    if tr.ndim == 4: tr = tr[0]
    
    # Take the first frame [0] and repeat it 49 times
    tr_static = np.tile(tr[0:1], (49, 1, 1))
    
    vis_mask = track["vis"]
    if vis_mask.ndim == 4: vis_mask = vis_mask[0]
    
    on_screen = (tr_static[..., 0] >= 0) & (tr_static[..., 1] >= 0) & (tr_static[..., 0] < track["W"]) & (tr_static[..., 1] < track["H"])
    
    vis_mask_bool = vis_mask.astype(bool)
    if vis_mask_bool.shape[-1] == 1: vis_mask_bool = vis_mask_bool[..., 0]
    
    vis_mask_static = np.tile(vis_mask_bool[0:1], (49, 1))
    
    final_vis = (vis_mask_static & on_screen)[..., None]

    # 3. Visualize
    vis.visualize(video=frames, tracks=tr_static, visibility=final_vis, filename="overlay", save_video=True)
    
    return output_vis_path

def get_ref_elements(ref_folder, video_basename):
    """
    Scans the ref folder for pairs of PNG images and NPY files.
    Returns a list of tuples: [(path_to_img_video, path_to_track_video), ...]
    """
    if not os.path.exists(ref_folder):
        print(f"Warning: Ref folder not found at {ref_folder}")
        return []

    files = os.listdir(ref_folder)
    files.sort()
    
    results = []
    
    # Identify unique base names that have a .png
    png_files = [f for f in files if f.endswith(".png")]
    
    # Temp folder for the image videos
    tmp_img_folder = os.path.join("./../tmp_webpage/tmp_custom_sig", video_basename + "_ref_imgs")
    os.makedirs(tmp_img_folder, exist_ok=True)
    
    for png_file in png_files:
        base_name = os.path.splitext(png_file)[0]
        npy_file = base_name + ".npy"
        
        # 1. Process Reference IMAGE -> Video
        try:
            fr = Image.open(os.path.join(ref_folder, png_file)).resize((832, 480))
            frames = np.array([np.array(fr)] * 49)
            
            img_video_path = os.path.join(tmp_img_folder, base_name + ".png") 
            save_video_jordan(img_video_path, frames, fps=15, quality=9)
            
        except Exception as e:
            print(f"Failed to process ref image {png_file}: {e}")
            continue

        # 2. Process Reference TRACK -> Video
        # Look for the specific .npy file in the same ref folder
        ref_npy_path = os.path.join(ref_folder, npy_file)
        
        if os.path.exists(ref_npy_path):
            # Generate visualization specifically for this reference npy
            # Suffix includes base_name to make it unique per reference (e.g. ref_001_track)
            track_video_path = get_static_vis_video(ref_npy_path, video_basename, suffix=f"ref_{base_name}")
        else:
            print(f"Warning: No matching .npy found for {png_file}, skipping track viz.")
            track_video_path = None
            
        results.append((img_video_path, track_video_path))
        
    return results

def get_vis_video(track_npy_path, video_basename, suffix="camera"):
    """
    Generates the MOVING Track visualization.
    Used for Source Track and Edited Track.
    """
    tmp_folder = os.path.join("./../tmp_webpage/tmp_vis_custom", video_basename + "_" + suffix)
    os.makedirs(tmp_folder, exist_ok=True)
    
    output_vis_path = os.path.join(tmp_folder, "tracking/overlay_tracking.mp4")
    
    if os.path.exists(output_vis_path):
        return output_vis_path
    
    if not os.path.exists(track_npy_path):
        print(f"Warning: Track file not found at {track_npy_path}")
        return None

    track = np.load(track_npy_path, allow_pickle=True).item()
    
    frames = np.zeros((49, track["H"], track["W"], 3), dtype=np.uint8)
    vis = Visualizer(save_dir=tmp_folder, linewidth=2, mode="rainbow", fps=15, tracks_leave_trace=0)

    tr = track["uvz"]
    if tr.ndim == 4: tr = tr[0]
    tr = tr[:49]
    
    vis_mask = track["vis"]
    if vis_mask.ndim == 4: vis_mask = vis_mask[0]
    
    on_screen = (tr[..., 0] >= 0) & (tr[..., 1] >= 0) & (tr[..., 0] < track["W"]) & (tr[..., 1] < track["H"])
    vis_mask_bool = vis_mask.astype(bool)
    if vis_mask_bool.shape[-1] == 1: vis_mask_bool = vis_mask_bool[..., 0]
        
    final_vis = (vis_mask_bool & on_screen)[..., None]

    vis.visualize(video=frames, tracks=tr, visibility=final_vis, filename="overlay", save_video=True)
    
    return output_vis_path

def get_source_video(subfolder_path, track_npy_path, video_basename):
    """
    Looks for source video at {subfolder}/tmp/video.mp4 or video720.mp4
    Returns a black placeholder if not found.
    """
    valid_names = ["video.mp4", "video720.mp4"]
    
    for fname in valid_names:
        potential_path = os.path.join(subfolder_path, "tmp", fname)
        if os.path.exists(potential_path):
            print(f"Found source video: {potential_path}")
            return potential_path

    print(f"Warning: Could not find video.mp4 or video720.mp4 in {subfolder_path}/tmp. Using black placeholder.")
    
    # Create Placeholder
    tmp_folder = os.path.join("./../tmp_webpage/tmp_source_placeholder", video_basename)
    os.makedirs(tmp_folder, exist_ok=True)
    output_path = os.path.join(tmp_folder, "source_placeholder.mp4")
    
    if os.path.exists(output_path):
        return output_path
        
    if not os.path.exists(track_npy_path):
        return None
        
    track = np.load(track_npy_path, allow_pickle=True).item()
    frames = np.zeros((49, track["H"], track["W"], 3), dtype=np.uint8)
    save_video_jordan(output_path, frames, fps=15, quality=9)
    return output_path

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile restylization results into stacked videos.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder.")
    parser.add_argument("--output_folder", type=str, default="./camera_custom_sig", help="Path to save output.")
    
    args = parser.parse_args()
    
    INPUT_FOLDER = args.input_folder
    OUTPUT_FOLDER = args.output_folder

    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist.")
        exit()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    video_extensions = ('.mp4', '.gif')
    files = os.listdir(INPUT_FOLDER)
    video_files = [f for f in files if f.lower().endswith(video_extensions)]

    print(f"Found {len(video_files)} videos to process in {INPUT_FOLDER}")

    for video_file in video_files:
        print(f"Processing {video_file}...")
        
        video_path = os.path.join(INPUT_FOLDER, video_file)
        video_basename = os.path.splitext(video_file)[0]
        subfolder_path = os.path.join(INPUT_FOLDER, video_basename)
        ref_folder = os.path.join(subfolder_path, "ref")
        track_npy_path = os.path.join(subfolder_path, "tmp", "track.npy")
        
        if not os.path.exists(subfolder_path):
            print(f"Skipping {video_file}: Corresponding folder '{subfolder_path}' not found.")
            continue

        # 1. Get Elements
        
        # A. Reference Pairs (Image + Track from .npy)
        ref_pairs = get_ref_elements(ref_folder, video_basename)
        
        # B. Moving Track Visualization (For Source Track & Edited Track)
        moving_track_path = get_vis_video(track_npy_path, video_basename, suffix="moving_track")
        
        # C. Source Video (Checks tmp/video.mp4 OR video720.mp4)
        source_video_path = get_source_video(subfolder_path, track_npy_path, video_basename)
        
        # 2. Build the Grid Dictionary
        inputs = {}
        
        # --- ROW 1 ---
        # Col 1: Source Track (Moving)
        inputs["Source Track"] = moving_track_path
        
        # Col 2..N: Reference Tracks (Specific Static Track from Ref NPY)
        for i, pair in enumerate(ref_pairs):
            ref_track_p = pair[1]
            if ref_track_p is None: ref_track_p = moving_track_path 
            inputs[f"Reference Track {i+1}"] = ref_track_p
            
        # Col Last: Edited Video Track (Moving)
        inputs["Edited Video Track"] = moving_track_path
        
        # --- ROW 2 ---
        # Col 1: Source Video (Bottom Left)
        inputs["Source Video"] = source_video_path
        
        # Col 2..N: Reference Images (Static)
        for i, pair in enumerate(ref_pairs):
            ref_img_p = pair[0]
            inputs[f"Ref Image {i+1}"] = ref_img_p
            
        # Col Last: Generated Video (Moving)
        inputs["Generated Video"] = video_path

        # 3. Process
        expected_cols = 1 + len(ref_pairs) + 1 # Source + Refs + Result
        
        if len(inputs) > 0:
            output_video_path = os.path.join(OUTPUT_FOLDER, video_basename + ".mp4")
            print(f"Stacking {len(inputs)} clips into {expected_cols} columns...")
            
            try:
                # Removed label_colors argument
                stack_videos.stack_videos(inputs, output_video_path, cols=expected_cols)
                print(f"Saved: {output_video_path}")
            except Exception as e:
                print(f"Failed to stack {video_file}: {e}")
        else:
            print(f"Not enough inputs generated for {video_basename}.")

    print("Done.")