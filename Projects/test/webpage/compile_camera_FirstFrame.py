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
from moviepy.editor import ImageSequenceClip
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

        clip = ImageSequenceClip(video_list, fps=self.fps)
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

def get_ref_imgs(ref_folder, video_basename):
    if not os.path.exists(ref_folder):
        print(f"Warning: Ref folder not found at {ref_folder}")
        return []

    ref_lis = os.listdir(ref_folder)
    ref_lis.sort()
    path_list = []
    
    tmp_folder = os.path.join("./../tmp_webpage/tmp_custom_sig", video_basename + "_ref")
    os.makedirs(tmp_folder, exist_ok=True)
    
    for r in ref_lis:
        if not r.endswith(".png"):
            continue
            
        try:
            fr = Image.open(os.path.join(ref_folder, r)).resize((832, 480))
        except Exception as e:
            print(f"Failed to open image {r}: {e}")
            continue

        frames = [np.array(fr)] * 49
        frames = np.array(frames)
        
        ref_idx = r.split(".png")[0]
        tmp_file = os.path.join(tmp_folder, ref_idx + ".png") 
        
        save_video_jordan(tmp_file, frames, fps=15, quality=9)
        path_list.append(tmp_file)
        
    return path_list

def get_vis_video(track_npy_path, video_basename):
    tmp_folder = os.path.join("./../tmp_webpage/tmp_vis_custom", video_basename + "_camera")
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

        inputs = {}
        
        # 1. Get Reference Videos
        path_list = get_ref_imgs(ref_folder, video_basename)
        names = ["First frame", "Second frame", "Third frame", "Fourth frame"]
        for ii in range(len(path_list)):
            label = names[ii] if ii < len(names) else f"Ref frame {ii}"
            inputs[label] = path_list[ii]

        # 2. Get Tracking Visualization
        vis_path = get_vis_video(track_npy_path, video_basename)
        if vis_path:
            inputs["Track conditions"] = vis_path

        # 3. Add Generated Video
        inputs["Generated video"] = video_path

        # NOTE: Removed the "Spacer" (White Video) generation block here.

        # 4. Stack and Save
        output_video_path = os.path.join(OUTPUT_FOLDER, video_basename + ".mp4")
        
        # DYNAMIC COLUMNS: Set cols to len(inputs) to avoid padding
        num_inputs = len(inputs)
        
        if num_inputs > 0:
            print(f"Stacking {num_inputs} videos for {video_basename}...")
            try:
                # Assuming stack_videos handles horizontal layout if cols == count
                stack_videos.stack_videos(inputs, output_video_path, cols=num_inputs)
                print(f"Saved: {output_video_path}")
            except Exception as e:
                print(f"Failed to stack {video_file}: {e}")
        else:
            print(f"No inputs found for {video_basename}, skipping.")

    print("Done.")