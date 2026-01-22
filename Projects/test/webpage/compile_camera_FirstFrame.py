"""
Compile restylization results
Usage:
    python compile_camera_FirstFrame.py --input_folder "/mnt/c/Users/koich/Siggraph/web/static_camera_custom" --output_folder "./camera_custom_sig"
    python compile_camera_FirstFrame.py --input_folder "/mnt/c/Users/koich/Siggraph/web/davis_static" --output_folder "./camera_custom_sig"
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

def get_first_ref_video(ref_folder, video_basename):
    """
    Finds the first PNG in the ref folder, repeats it for the video duration,
    and returns the path to the static video.
    """
    if not os.path.exists(ref_folder):
        print(f"Warning: Ref folder not found at {ref_folder}")
        return None

    files = os.listdir(ref_folder)
    files.sort()
    
    first_png = None
    for f in files:
        if f.endswith(".png"):
            first_png = f
            break
            
    if not first_png:
        print(f"Warning: No png found in {ref_folder}")
        return None
        
    tmp_folder = os.path.join("./../tmp_webpage/tmp_custom_sig", video_basename + "_ref_imgs")
    os.makedirs(tmp_folder, exist_ok=True)
    
    try:
        fr = Image.open(os.path.join(ref_folder, first_png)).resize((832, 480))
        frames = np.array([np.array(fr)] * 49)
        output_path = os.path.join(tmp_folder, "first_frame.mp4")
        save_video_jordan(output_path, frames, fps=15, quality=9)
        return output_path
        
    except Exception as e:
        print(f"Failed to process first ref image {first_png}: {e}")
        return None

def get_vis_video(track_npy_path, video_basename, suffix="camera"):
    """
    Generates the MOVING Track visualization.
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

def get_black_video(video_basename):
    """Generates a 49-frame black spacer video (832x480)."""
    tmp_folder = os.path.join("./../tmp_webpage/tmp_spacers", video_basename)
    os.makedirs(tmp_folder, exist_ok=True)
    output_path = os.path.join(tmp_folder, "black.mp4")
    
    if os.path.exists(output_path):
        return output_path
        
    # Standard dimensions used in script
    frames = np.zeros((49, 480, 832, 3), dtype=np.uint8)
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
    
    # 1. Collect and Deduplicate Files
    all_files = sorted(os.listdir(INPUT_FOLDER))
    unique_files_map = {} 
    
    for f in all_files:
        if f.lower().endswith(('.mp4', '.gif')):
            base_name = os.path.splitext(f)[0]
            ext = os.path.splitext(f)[1].lower()
            if base_name not in unique_files_map:
                unique_files_map[base_name] = f
            else:
                if ext == '.mp4' and not unique_files_map[base_name].lower().endswith('.mp4'):
                    unique_files_map[base_name] = f

    # 2. Group Files by Prefix
    grouped_files = {} 
    for base_name, full_name in unique_files_map.items():
        if '_' in base_name:
            prefix = base_name.rsplit('_', 1)[0]
        else:
            prefix = base_name
            
        if prefix not in grouped_files:
            grouped_files[prefix] = []
        grouped_files[prefix].append(full_name)

    print(f"Found {len(grouped_files)} unique groups to process.")

    # 3. Process Groups
    for prefix, video_files in grouped_files.items():
        video_files.sort()
        print(f"Processing group '{prefix}' with {len(video_files)} variants...")
        
        pairs = [] # List of tuples: (Track, GeneratedVideo)
        
        ref_video_path = None
        black_video_path = None
        
        # A. Collect all pairs first
        for video_file in video_files:
            video_path = os.path.join(INPUT_FOLDER, video_file)
            video_basename = os.path.splitext(video_file)[0]
            subfolder_path = os.path.join(INPUT_FOLDER, video_basename)
            ref_folder = os.path.join(subfolder_path, "ref")
            track_npy_path = os.path.join(subfolder_path, "tmp", "track.npy")
            
            if not os.path.exists(subfolder_path):
                print(f"Skipping {video_file}: Corresponding folder '{subfolder_path}' not found.")
                continue

            # Grab reference/black video if we haven't yet
            if ref_video_path is None:
                ref_video_path = get_first_ref_video(ref_folder, video_basename)
                if ref_video_path:
                    black_video_path = get_black_video(video_basename)

            track_vis_path = get_vis_video(track_npy_path, video_basename, suffix="track_cond")
            
            if track_vis_path and video_path:
                pairs.append((track_vis_path, video_path))
            else:
                print(f"Missing track or video for {video_file}, skipping.")

        # B. Construct the Grid DICTIONARY
        # Desired Layout (5 Cols):
        # Row 1: Ref | T1 | V1 | T2 | V2
        # Row 2: Blk | T3 | V3 | T4 | V4
        
        if len(pairs) > 0 and ref_video_path and black_video_path:
            grid_inputs = {}
            
            num_pairs = len(pairs)
            
            # Using row_idx to create unique keys containing spaces
            row_idx = 1
            
            for i in range(0, num_pairs, 2):
                
                # --- Col 1: Reference or Spacer (Empty Label) ---
                if i == 0:
                    grid_inputs[f"First frame"] = ref_video_path
                else:
                    # Key must be unique, so we use increasing spaces
                    # " " * 1, " " * 2, etc. These render as empty strings visually.
                    unique_space_key = " " * row_idx
                    grid_inputs[unique_space_key] = black_video_path
                
                # --- Cols 2 & 3: Pair 1 ---
                p1 = pairs[i]
                grid_inputs[f"Track Conditions {i+1}"] = p1[0]
                grid_inputs[f"Generated Video {i+1}"] = p1[1]
                
                # --- Cols 4 & 5: Pair 2 OR Spacer (Empty Label) ---
                if i + 1 < num_pairs:
                    p2 = pairs[i+1]
                    grid_inputs[f"Track Conditions {i+2}"] = p2[0]
                    grid_inputs[f"Generated Video {i+2}"] = p2[1]
                else:
                    # Pad with black if odd number of pairs
                    # Ensure keys are unique by adding offset to spaces
                    unique_space_key_t = " " * (row_idx + 100)
                    unique_space_key_v = " " * (row_idx + 200)
                    grid_inputs[unique_space_key_t] = black_video_path
                    grid_inputs[unique_space_key_v] = black_video_path
                    
                row_idx += 1

            output_video_path = os.path.join(OUTPUT_FOLDER, prefix + ".mp4")
            expected_cols = 5
            
            print(f"Stacking {len(grid_inputs)} items into {expected_cols} cols for group {prefix}...")
            
            try:
                stack_videos.stack_videos(grid_inputs, output_video_path, cols=expected_cols)
                print(f"Saved: {output_video_path}")
            except Exception as e:
                print(f"Failed to stack group {prefix}: {e}")
                
        else:
            print(f"Skipping group {prefix}: Missing reference frame or pairs.")

    print("Done.")