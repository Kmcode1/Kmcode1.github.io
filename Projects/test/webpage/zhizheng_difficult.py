"""
Compile restylization results: Individual Videos with Smart Vertical Layout
Usage:
     python zhizheng_difficult.py --input_folder "/mnt/c/Users/koich/Siggraph/web/zhizheng_static_difficult" --output_folder "./static_z"
"""

import os
import sys
import argparse
import numpy as np
import cv2
import imageio
from PIL import Image
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import time

# Local import
sys.path.append(".")
import stack_videos

def save_video_jordan(output_path, video, fps, quality=None, imageio_params=None):
    imageio_params = imageio_params if imageio_params is not None else {}
    if quality is not None:
        imageio_params["quality"] = quality
    if os.path.splitext(output_path)[1].lower() == ".gif":
        imageio_params["loop"] = 0

    try:
        writer = imageio.get_writer(output_path, fps=fps, **imageio_params)
        for i in range(len(video)):
            writer.append_data(np.array(video[i]))
        writer.close()
    except Exception as e:
        print(f"Error saving video {output_path}: {e}")

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
            writer = imageio.get_writer(save_path, fps=self.fps, codec="libx264", quality=8)
            for frame in video_list:
                writer.append_data(frame)
            writer.close()
            time.sleep(0.1)
        except Exception as e:
            print(f"Failed to write visualization {save_path}: {e}")

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

        for t in tqdm(range(T), desc=f"Drawing {filename}"):
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
                if t == 0:
                    diffs = tracks[0] - tracks[0, i]
                    distances = np.linalg.norm(diffs, axis=1)
                    valid_dists = distances[distances > 0]
                    rect_size = (int(np.min(valid_dists)) / 2) if len(valid_dists) > 0 else 5
                
                if is_visible:
                    cv2.circle(res_video[t], (int(coord[0]), int(coord[1])), int(self.linewidth * 2), vector_colors[t, i].tolist(), thickness=-1)

        return np.stack(res_video)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_blank_video():
    """Generates a black blank video for padding track rows."""
    path = "./../tmp_webpage/black_padding.mp4"
    if os.path.exists(path):
        return path
    
    os.makedirs("./../tmp_webpage", exist_ok=True)
    frames = (np.zeros((49, 480, 832, 3))).astype(np.uint8) 
    save_video_jordan(path, frames, fps=15, quality=9)
    return path

def is_valid_video(path):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return False
        ret, _ = cap.read()
        cap.release()
        return ret
    except:
        return False

def get_vis_video(track_npy_path, video_basename, suffix_id, background_frames=None, width=None, height=None):
    tmp_folder = os.path.join("./../tmp_webpage/tmp_vis_individual", video_basename)
    os.makedirs(tmp_folder, exist_ok=True)
    
    filename_base = f"vis_{suffix_id}"
    expected_filename = f"{filename_base}_tracking.mp4"
    output_vis_path = os.path.join(tmp_folder, "tracking", expected_filename)
    
    if os.path.exists(output_vis_path):
        if is_valid_video(output_vis_path):
            return output_vis_path
        else:
            print(f"Regenerating corrupted: {output_vis_path}")
            os.remove(output_vis_path)
    
    if not os.path.exists(track_npy_path):
        return None

    try:
        track = np.load(track_npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading {track_npy_path}: {e}")
        return None
    
    H = height if height else track.get("H", 480)
    W = width if width else track.get("W", 832)
    
    if background_frames is not None:
        frames = background_frames
    else:
        frames = np.zeros((49, H, W, 3), dtype=np.uint8)
    
    vis = Visualizer(save_dir=tmp_folder, linewidth=2, mode="rainbow", fps=15, tracks_leave_trace=0)

    tr = track["uvz"]
    if tr.ndim == 4: tr = tr[0]
    tr = tr[:49]
    vis_mask = track["vis"]
    if vis_mask.ndim == 4: vis_mask = vis_mask[0]
    
    if tr.shape[0] != frames.shape[0]:
        min_len = min(tr.shape[0], frames.shape[0])
        tr = tr[:min_len]
        vis_mask = vis_mask[:min_len]
        frames = frames[:min_len]

    on_screen = (tr[..., 0] >= 0) & (tr[..., 1] >= 0) & (tr[..., 0] < W) & (tr[..., 1] < H)
    vis_mask_bool = vis_mask.astype(bool)
    if vis_mask_bool.shape[-1] == 1: vis_mask_bool = vis_mask_bool[..., 0]
    final_vis = (vis_mask_bool & on_screen)[..., None]

    vis.visualize(video=frames, tracks=tr, visibility=final_vis, filename=filename_base, save_video=True)
    return output_vis_path

def get_ref_data(ref_folder, video_basename):
    """Returns list of tuples: (image_path, track_path_or_None)"""
    if not os.path.exists(ref_folder): return []

    ref_files = os.listdir(ref_folder)
    indices = set()
    for f in ref_files:
        if f.endswith(".png"):
            try:
                indices.add(int(f.split(".png")[0]))
            except: pass
    
    sorted_indices = sorted(list(indices))
    
    data_pairs = []
    
    tmp_folder = os.path.join("./../tmp_webpage/tmp_custom_individual", video_basename + "_ref")
    os.makedirs(tmp_folder, exist_ok=True)
    
    for idx in sorted_indices:
        png_name = f"{idx}.png"
        npy_name = f"{idx}.npy"
        
        png_full = os.path.join(ref_folder, png_name)
        npy_full = os.path.join(ref_folder, npy_name)
        
        try:
            fr = Image.open(png_full).resize((832, 480))
            frame_arr = np.array(fr)
            frames = np.array([frame_arr] * 49) 
            
            img_vid_path = os.path.join(tmp_folder, f"ref_img_{idx}.mp4")
            if not os.path.exists(img_vid_path):
                save_video_jordan(img_vid_path, frames, fps=15, quality=9)
            
            track_vid_path = None
            if os.path.exists(npy_full):
                # No overlay, black background
                track_vid_path = get_vis_video(
                    npy_full, video_basename, 
                    suffix_id=f"ref_{idx}", 
                    background_frames=None, 
                    width=832, height=480
                )
            
            data_pairs.append((img_vid_path, track_vid_path))

        except Exception as e:
            print(f"Error processing ref {idx}: {e}")
            continue
            
    return data_pairs

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile restylization results: Hybrid Layout.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input.")
    parser.add_argument("--output_folder", type=str, default="./camera_individual", help="Path to output.")
    
    args = parser.parse_args()
    
    INPUT_FOLDER = args.input_folder
    OUTPUT_FOLDER = args.output_folder

    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist.")
        exit()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 1. Scan and Prioritize Files
    all_files = os.listdir(INPUT_FOLDER)
    file_map = {} 
    
    for f in all_files:
        if f.startswith("."): continue
        name, ext = os.path.splitext(f)
        ext = ext.lower()
        if ext not in ['.mp4', '.gif']: continue
        
        if name in file_map:
            curr_ext = os.path.splitext(file_map[name])[1].lower()
            if curr_ext == '.gif' and ext == '.mp4':
                file_map[name] = f
        else:
            file_map[name] = f

    valid_files = sorted(list(file_map.values()))
    print(f"Found {len(valid_files)} unique videos.")
    
    blank_video = get_blank_video()

    for video_file in valid_files:
        print(f"Processing {video_file}...")
        
        basename = os.path.splitext(video_file)[0]
        video_path = os.path.join(INPUT_FOLDER, video_file)
        
        subfolder_path = os.path.join(INPUT_FOLDER, basename)
        ref_path = os.path.join(subfolder_path, "ref")
        track_path = os.path.join(subfolder_path, "tmp", "track.npy")
        
        inputs = {}
        
        # A. Collect Units (Img, Track)
        # We treat every entity (Reference 1, Reference 2, Generated Video) as a "Unit"
        # Each Unit consists of (Track, Image/Video)
        units = []
        
        # Add References
        ref_pairs = get_ref_data(ref_path, basename)
        units.extend(ref_pairs)
        
        # Add Result
        res_trk_path = get_vis_video(track_path, basename, suffix_id="main", background_frames=None)
        units.append((video_path, res_trk_path))
        
        # B. Decide Layout
        # Special Case: EXACTLY 2 Units (e.g., 1 Ref + 1 Result = 4 videos total)
        # User wants them all in ONE row.
        # Order: [Ref Trk] [Ref Img] [Vid Trk] [Vid Res]
        
        if len(units) == 2:
            cols = 4
            
            # Unit 1 (Reference)
            img_p, trk_p = units[0]
            inputs["00_Ref_Track"] = trk_p if trk_p else blank_video
            inputs["01_Ref_Image"] = img_p
            
            # Unit 2 (Result)
            vid_p, res_trk_p = units[1]
            inputs["02_Vid_Track"] = res_trk_p if res_trk_p else blank_video
            inputs["03_Vid_Result"] = vid_p
            
        else:
            # General Case: Standard 2-Row Layout
            # Row 1 (Top): Tracks
            # Row 2 (Bot): Images
            cols = len(units)
            
            for i, (img_p, trk_p) in enumerate(units):
                # Naming keys to enforce: Row 00 (Top), Row 01 (Bot), Column i
                
                # Top Row: Track
                key_top = f"00_{i:02d}_Track"
                inputs[key_top] = trk_p if trk_p else blank_video
                
                # Bottom Row: Image/Video
                key_bot = f"01_{i:02d}_Image"
                inputs[key_bot] = img_p

        output_filename = video_file.replace(".gif", ".mp4")
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        try:
            stack_videos.stack_videos(inputs, output_path, cols=cols)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Failed to stack {basename}: {e}")

    print("Done.")