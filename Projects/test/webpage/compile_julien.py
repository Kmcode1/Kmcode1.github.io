"""
Compile restylization results
conda activate das
cd /root/Netflix/myproject/data_preprocess/webpage/
rm -r videos_jul
python compile_julien.py

Input:
/fsx_scanline/from_eyeline/users/jphilip/test_delighting/koichi_cvpr26
13470978_1280_720_24fps        3127085-hd_1280_720_24fps      3326746-hd_1280_720_24fps      3770033-hd_1280_720_25fps
13470978_1280_720_24fps.mp4    3127085-hd_1280_720_24fps.mp4  3326746-hd_1280_720_24fps.mp4  3770033-hd_1280_720_25fps.mp4
3064220-hd_1280_720_24fps      3135811-hd_1280_720_24fps      3327058-hd_1280_720_24fps      7578544-hd_1280_720_30fps
3064220-hd_1280_720_24fps.mp4  3135811-hd_1280_720_24fps.mp4  3327058-hd_1280_720_24fps.mp4  7578544-hd_1280_720_30fps.mp4

/fsx_scanline/from_eyeline/users/jphilip/test_delighting/koichi_cvpr26/13470978_1280_720_24fps/frame_00001.png
    - frame_00001.png
"""

import torch, os, json, sys
sys.path.append(".")
import stack_videos
import argparse
from pathlib import Path
import random
import os
import subprocess
from decord import VideoReader, cpu
import mediapy as media
import numpy as np
from PIL import Image
import PIL
import cv2
import imageio
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Added for better text support

from PIL import Image
from io import BytesIO
import os
import base64
import types
import time
from decord import VideoReader, cpu
import numpy as np
import mediapy as media


import fsspec
from contextlib import contextmanager
import tempfile
import imageio
import subprocess

import os
import numpy as np
import cv2
import torch
import flow_vis


import random
from datetime import datetime

from matplotlib import cm
import torch.nn.functional as F
import torchvision.transforms as transforms
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
from tqdm import tqdm

import OpenEXR
import Imath
import numpy as np
import sys
import cv2
def read_exr(file_path):
    """
    Read an EXR file and return the image data as a numpy array.
    
    Args:
        file_path (str): Path to the EXR file
        
    Returns:
        numpy.ndarray: Image data
    """
    # Open the EXR file
    exr_file = OpenEXR.InputFile(file_path)
    
    # Get the header
    header = exr_file.header()
    
    # Get image dimensions
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Get channel names
    channels = header['channels'].keys()
    print(f"Available channels: {list(channels)}")
    
    # Read RGB channels (or available channels)
    channel_data = {}
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    
    for channel in ['R', 'G', 'B']:
        if channel in channels:
            channel_str = exr_file.channel(channel, pixel_type)
            channel_data[channel] = np.frombuffer(channel_str, dtype=np.float32)
            channel_data[channel] = channel_data[channel].reshape((height, width))
    
    # Stack channels into a single array
    if len(channel_data) == 3:
        image = np.stack([channel_data['R'], channel_data['G'], channel_data['B']], axis=2)
    elif len(channel_data) == 1:
        image = list(channel_data.values())[0]
    else:
        # Read first available channel
        first_channel = list(channels)[0]
        channel_str = exr_file.channel(first_channel, pixel_type)
        image = np.frombuffer(channel_str, dtype=np.float32).reshape((height, width))
    
    return image, header

def Lin_to_Log(im,max_val=128):
    im = np.clip(im,max=max_val)
    im = np.log(2.2*im + 1.0)/np.log(2.2*max_val + 1.0)
    return np.power(im,1.0/2.2)

def Lin_to_sRGB(im):
    """Convert linear to sRGB."""
    linear_part = 12.92 * im
    gamma_part = im ** (1.0 / 2.4) * 1.055 - 0.055
    return np.where(im <= 0.0031308, linear_part, gamma_part)


def save_video_jordan(output_path, video, fps, quality=None, imageio_params=None, save_individual = False):
    """
    Args:
        video: F x H x W x 3, in np.uint8 (so 0-255)
    
    Jordan:
    
    My function for saving mp4 and gifs (depending on the output name you give it, it will automatically figure out if it needs to save a gif or an mp4):
First, install imageio with pip install "imageio[ffmpeg]", then ffmpeg is important because otherwise it will not support GIF saving
    
    Paul (https://www.reddit.com/r/photoshop/comments/h8uiq3/cant_seem_to_get_accurate_30_fps_in_photoshop/):
    
    GIF files that contain animation do not store a "framerate". Instead it stores, per frame, a "frame duration". And that frame duration is stored as an integer (n) that defines n/100ths of a second. So you can specify a frame duration of 1/100 sec, 2/100 sec, 3/100 sec, 4/100 sec, etc. but not something like 3.3333.../100 sec. (30 fps).
    you can not get 60 fps. But you can get 25 and 50 fps.
    """
    
    imageio_params = imageio_params if imageio_params is not None else {}
    if quality is not None:  # Quality is 1-10, generally for MP4 I do 7 or above, unsure how quality affects GIFs
        imageio_params["quality"] = quality
    if os.path.splitext(output_path)[1] == ".gif":  # Makes sure the gif will loop when it ends
        imageio_params["loop"] = 0

    if True:
        writer = imageio.get_writer(output_path, fps=fps, **imageio_params)
        for i in range(len(video)):
            writer.append_data(np.array(video[i]))
        writer.close()

def read_video_from_path(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
    else:
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                break
        cap.release()
    return np.stack(frames)


class Visualizer:
    def __init__(
        self,
        save_dir: str = "./results",
        grayscale: bool = False,
        pad_value: int = 0,
        fps: int = 10,
        mode: str = "rainbow",  # 'cool', 'optical_flow'
        linewidth: int = 1,
        show_first_frame: int = 10,
        tracks_leave_trace: int = 0,  # -1 for infinite
    ):
        self.mode = mode
        self.save_dir = save_dir
        self.vtxt_path = os.path.join(save_dir, "videos.txt")
        self.ttxt_path = os.path.join(save_dir, "trackings.txt")
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
        self.show_first_frame = show_first_frame
        self.grayscale = grayscale
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps

    def visualize(
        self,
        video: torch.Tensor,  # (B,T,C,H,W)
        tracks: torch.Tensor,  # (B,T,N,2)
        visibility: torch.Tensor = None,  # (B, T, N, 1) bool
        gt_tracks: torch.Tensor = None,  # (B,T,N,2)
        segm_mask: torch.Tensor = None,  # (B,1,H,W)
        filename: str = "video",
        writer=None,  # tensorboard Summary Writer, used for visualization during training
        step: int = 0,
        query_frame: int = 0,
        save_video: bool = True,
        compensate_for_camera_motion: bool = False,
        rigid_part = None,
        video_depth = None # (B,T,C,H,W)
    ):
        if compensate_for_camera_motion:
            assert segm_mask is not None
        if segm_mask is not None:
            coords = tracks[0, query_frame].round().long()
            segm_mask = segm_mask[0, query_frame][coords[:, 1], coords[:, 0]].long()

        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )

        if video_depth is not None:
            video_depth = (video_depth*255).cpu().numpy().astype(np.uint8)
            video_depth = ([cv2.applyColorMap(video_depth[0,i,0], cv2.COLORMAP_INFERNO) 
                            for i in range(video_depth.shape[1])])
            video_depth = np.stack(video_depth, axis=0)
            video_depth = torch.from_numpy(video_depth).permute(0, 3, 1, 2)[None]

        tracks = tracks + self.pad_value

        if self.grayscale:
            transform = transforms.Grayscale()
            video = transform(video)
            video = video.repeat(1, 1, 3, 1, 1)

        tracking_video = self.draw_tracks_on_video(
            video=video,
            tracks=tracks,
            visibility=visibility,
            segm_mask=segm_mask,
            gt_tracks=gt_tracks,
            query_frame=query_frame,
            compensate_for_camera_motion=compensate_for_camera_motion,
            rigid_part=rigid_part
        )

        if save_video:
            # import ipdb; ipdb.set_trace()
            tracking_dir = os.path.join(self.save_dir, "tracking")
            if not os.path.exists(tracking_dir):
                os.makedirs(tracking_dir)
            self.save_video(tracking_video, filename=filename+"_tracking", 
                            savedir=tracking_dir, writer=writer, step=step)
            # with open(self.ttxt_path, 'a') as file:
            #     file.write(f"tracking/{filename}_tracking.mp4\n")

            videos_dir = os.path.join(self.save_dir, "videos")
            if not os.path.exists(videos_dir):
                os.makedirs(videos_dir)
            self.save_video(video, filename=filename, 
                            savedir=videos_dir, writer=writer, step=step)
            
        return tracking_video

    def save_video(self, video, filename, savedir=None, writer=None, step=0):
        if writer is not None:
            writer.add_video(
                f"{filename}",
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
            # clip = ImageSequenceClip(wide_list[2:-1], fps=self.fps)
            clip = ImageSequenceClip(wide_list, fps=self.fps)

            # Write the video file
            if savedir is None:
                save_path = os.path.join(self.save_dir, f"{filename}.mp4")
            else:
                save_path = os.path.join(savedir, f"{filename}.mp4")
            clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)

            print(f"Video saved to {save_path}")

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        gt_tracks=None,
        query_frame: int = 0,
        compensate_for_camera_motion=False,
        rigid_part=None,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 3
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].detach().cpu().numpy()  # S, N, 2
        if gt_tracks is not None:
            gt_tracks = gt_tracks[0].detach().cpu().numpy()

        res_video = []

        # process input video
        for rgb in video:
             res_video.append(rgb.copy())
        
        ## create a blank tensor with the same shape as the video
        #for rgb in video:
        #    black_frame = np.zeros_like(rgb.copy(), dtype=rgb.dtype)
        #    res_video.append(black_frame)

        vector_colors = np.zeros((T, N, 3))

        if self.mode == "optical_flow":

            vector_colors = flow_vis.flow_to_color(tracks - tracks[query_frame][None])

        elif segm_mask is None:
            if self.mode == "rainbow":
                x_min, x_max = tracks[0, :, 0].min(), tracks[0, :, 0].max()
                y_min, y_max = tracks[0, :, 1].min(), tracks[0, :, 1].max()

                if True:
                    x_min = 1111111111
                    x_max = -11111111111
                    y_min = 1111111111
                    y_max = -11111111111
                    """
                    visibility torch.Size([1, 49, 15000, 1])
                    f tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0])
                    tracks (49, 15000, 3)
                    """
                    for num_tracks in range(tracks.shape[2]):
                        f = (visibility[0, :, num_tracks,0] !=0).to(torch.float32).argmax(dim=0)
                        print("visibility2", visibility.shape)
                        print("f", f)
                        print("tracks", tracks.shape)
                        #f = -1
                        x_min = min(tracks[f, num_tracks, 0], x_min)
                        x_max = max(tracks[f, num_tracks, 0], x_max)
                        y_min = min(tracks[f, num_tracks, 1], y_min)
                        y_max = max(tracks[f, num_tracks, 1], y_max)

                z_inv = 1/tracks[0, :, 2]
                z_min, z_max = np.percentile(z_inv, [2, 98])
                
                norm_x = plt.Normalize(x_min, x_max)
                norm_y = plt.Normalize(y_min, y_max)
                norm_z = plt.Normalize(z_min, z_max)

                for n in range(N):
                    f = (visibility[0, :, n, 0] != 0).to(torch.float32).argmax(dim=0)
                    #f = 0
                    r = norm_x(tracks[f, n, 0])
                    g = norm_y(tracks[f, n, 1])
                    # r = 0
                    # g = 0
                    b = norm_z(1/tracks[0, n, 2])
                    color = np.array([r, g, b])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)
            else:
                # color changes with time
                for t in range(T):
                    color = np.array(self.color_map(t / T)[:3])[None] * 255
                    vector_colors[t] = np.repeat(color, N, axis=0)
        else:
            if self.mode == "rainbow":
                vector_colors[:, segm_mask <= 0, :] = 255

                x_min, x_max = tracks[0, :, 0].min(), tracks[0, :, 0].max()
                y_min, y_max = tracks[0, :, 1].min(), tracks[0, :, 1].max()
                z_min, z_max = tracks[0, :, 2].min(), tracks[0, :, 2].max()

                norm_x = plt.Normalize(x_min, x_max)
                norm_y = plt.Normalize(y_min, y_max)
                norm_z = plt.Normalize(z_min, z_max)

                for n in range(N):
                    r = norm_x(tracks[0, n, 0])
                    g = norm_y(tracks[0, n, 1])
                    b = norm_z(tracks[0, n, 2])
                    color = np.array([r, g, b])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)

            else:
                # color changes with segm class
                segm_mask = segm_mask.cpu()
                color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
                color[segm_mask > 0] = np.array(self.color_map(1.0)[:3]) * 255.0
                color[segm_mask <= 0] = np.array(self.color_map(0.0)[:3]) * 255.0
                vector_colors = np.repeat(color[None], T, axis=0)

        # Draw tracks
        if self.tracks_leave_trace != 0:
            for t in range(1, T):
                first_ind = (
                    max(0, t - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]
                if compensate_for_camera_motion:
                    diff = (
                        tracks[first_ind : t + 1, segm_mask <= 0]
                        - tracks[t : t + 1, segm_mask <= 0]
                    ).mean(1)[:, None]

                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]

                res_video[t] = self._draw_pred_tracks(
                    res_video[t],
                    curr_tracks,
                    curr_colors,
                )
                if gt_tracks is not None:
                    res_video[t] = self._draw_gt_tracks(
                        res_video[t], gt_tracks[first_ind : t + 1]
                    )

        if rigid_part is not None:
            cls_label = torch.unique(rigid_part)
            cls_num = len(torch.unique(rigid_part))
            # visualize the clustering results 
            cmap = plt.get_cmap('jet')  # get the color mapping
            colors = cmap(np.linspace(0, 1, cls_num))  
            colors = (colors[:, :3] * 255) 
            color_map = {lable.item(): color for lable, color in zip(cls_label, colors)}

        # Draw points
        for t in tqdm(range(T)):
            # Create a list to store information for each point
            points_info = []
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                depth = tracks[t, i, 2]  # assume the third dimension is depth
                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i]
                if coord[0] != 0 and coord[1] != 0:
                    if not compensate_for_camera_motion or (
                        compensate_for_camera_motion and segm_mask[i] > 0
                    ):
                        points_info.append((i, coord, depth, visibile))
            
            # Sort points by depth, points with smaller depth (closer) will be drawn later
            points_info.sort(key=lambda x: x[2], reverse=True)
            
            for i, coord, _, visibile in points_info:
                if rigid_part is not None:
                    color = color_map[rigid_part.squeeze()[i].item()]
                    cv2.circle(
                        res_video[t],
                        coord,
                        int(self.linewidth * 2),
                        color.tolist(),
                        thickness=-1 if visibile else 2
                        -1,
                    )
                else:
                    # Determine rectangle width based on the distance between adjacent tracks in the first frame
                    if t == 0:
                        distances = np.linalg.norm(tracks[0] - tracks[0, i], axis=1)
                        distances = distances[distances > 0]
                        rect_size = int(np.min(distances))/2
                    
                    # Define coordinates for top-left and bottom-right corners of the rectangle
                    top_left = (int(coord[0] - rect_size), int(coord[1] - rect_size/1.5)) # Rectangle width is 1.5x (video aspect ratio is 1.5:1)
                    bottom_right = (int(coord[0] + rect_size), int(coord[1] + rect_size/1.5))

                    # Draw rectangle
                    if visibile:
                        print("center", coord)
                        cv2.circle(
                            res_video[t],
                            (int(coord[0]), int(coord[1])),
                            int(self.linewidth * 2),
                            vector_colors[t, i].tolist(),
                            thickness=-1 if visibile else 0
                            -1,
                        )

        # Construct the final rgb sequence
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def _draw_pred_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x 2
        vector_colors: np.ndarray,
        alpha: float = 0.5,
    ):
        T, N, _ = tracks.shape

        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].tolist(),
                        self.linewidth,
                        cv2.LINE_AA,
                    )
            if self.tracks_leave_trace > 0:
                rgb = cv2.addWeighted(rgb, alpha, original, 1 - alpha, 0)
        return rgb

    def _draw_gt_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3,
        gt_tracks: np.ndarray,  # T x 2
    ):
        T, N, _ = gt_tracks.shape
        color = np.array((211.0, 0.0, 0.0))

        for t in range(T):
            for i in range(N):
                gt_tracks = gt_tracks[t][i]
                #  draw a red cross
                if gt_tracks[0] > 0 and gt_tracks[1] > 0:
                    length = self.linewidth * 3
                    coord_y = (int(gt_tracks[0]) + length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) - length, int(gt_tracks[1]) - length)
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                        cv2.LINE_AA,
                    )
                    coord_y = (int(gt_tracks[0]) - length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) + length, int(gt_tracks[1]) - length)
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                        cv2.LINE_AA,
                    )
        return rgb

good_examples = [#"13470978_1280_720_24fps_albedo_Original.gif",
                 #"3327058-hd_1280_720_24fps_albedo_Ref1_Original.mp4",
                 "13470978_1280_720_24fps_albedo_Ref2.mp4",
                 "3327058-hd_1280_720_24fps_albedo_Ref2.mp4", #must have
                 #"13470978_1280_720_24fps_albedo_Ref2_Original.mp4",
                 #"3327058-hd_1280_720_24fps_albedo_Ref2_Original.mp4",
                 "3327058-hd_1280_720_24fps_lighting_Ref2.mp4", #must have
                 #"3135811-hd_1280_720_24fps_albedo_Ref1_Original.mp4",
                 "3135811-hd_1280_720_24fps_albedo_Ref2.mp4",
                 "3770033-hd_1280_720_25fps_albedo_Ref2.mp4", #must have
                 #"3135811-hd_1280_720_24fps_albedo_Ref2_Original.mp4",
                 #3770033-hd_1280_720_25fps_albedo_Ref2_Original.mp4",
                 #"3135811-hd_1280_720_24fps_lighting.mp4",
                 "3770033-hd_1280_720_25fps_lighting_Ref2.mp4",
                 #"7578544-hd_1280_720_30fps_albedo_Original.mp4",
                 "3326746-hd_1280_720_24fps_albedo_Ref2.mp4", #must have
                 #"3326746-hd_1280_720_24fps_albedo_Ref2_Original.mp4",
                 "7578544-hd_1280_720_30fps_albedo_Ref2.mp4",
                 #"7578544-hd_1280_720_30fps_albedo_Ref2_Original.mp4",
                 "7578544-hd_1280_720_30fps_lighting_Ref2.mp4", #must have
                 #"3327058-hd_1280_720_24fps_albedo_Ref1_Original.gif"
                ]

for i in range(len(good_examples)):
    good_examples[i] = os.path.join("/root/Netflix/myproject/train/eval/julien_final_stylized/Netflix65K_14B_RandomVec30_ControlBefore_DecompQKVMLPNormQK_RefTimeNew_SelfQK_Lora8OV_1DTrack_PromptDrop15_Only49_3e5_AW3_DS_NoVAEOffload_Ref4_Crop0_W1000_Removal80_WanfunV3_22000/Netflix65K_14B_RandomVec30_ControlBefore_DecompQKVMLPNormQK_RefTimeNew_SelfQK_Lora8OV_1DTrack_PromptDrop15_Only49_3e5_AW3_DS_NoVAEOffload_Ref4_Crop0_W1000_Removal80_WanfunV3_22000/", good_examples[i].replace(".gif", ".mp4"))

def get_gt_video(path):
    lis = path.split("/")
    model_name = lis[-2]
    video_name = lis[-1]
    path = os.path.join("/".join(lis[:-2]), "log/", video_name.split(".mp4")[0], "tmp/video.mp4")
    return path

def get_naive_video(path):
    """
    Get naively generated videos
    """
    lis = path.split("/")
    model_name = lis[-2]
    video_name = lis[-1]
    if "Original" in path:
        path = os.path.join("/".join(lis[:-2]), "log/", video_name.split(".mp4")[0], "tmp/video.mp4")
        return path
    
    #enumerate reference images
    path_folder = None
    mode = "albedo"
    video_path = None

    tmp_file = "./../tmp_webpage/tmp_jul/" + mode + "_" + video_name
    os.makedirs(tmp_file, exist_ok = True)
    tmp_file = os.path.join(tmp_file, "video.mp4")
    if os.path.exists(tmp_file):
        return tmp_file
    
    if "albedo" in path:
        video_name = "_".join(video_name.split("_")[:4])
        path_folder = os.path.join("/fsx_scanline/from_eyeline/users/jphilip/test_delighting/koichi_cvpr26/", video_name, "output")
        video_path = os.path.join("/fsx_scanline/from_eyeline/users/jphilip/test_delighting/koichi_cvpr26/", video_name + ".mp4")
        mode = "albedo"
        #lighting_r_1_frame_00341.exr
        #olats_albedo_frame_00078.exr
    
    if "lighting" in path:
        video_name = "_".join(video_name.split("_")[:4])
        path_folder = os.path.join("/fsx_scanline/from_eyeline/users/jphilip/test_delighting/koichi_cvpr26/", video_name, "output")
        video_path = os.path.join("/fsx_scanline/from_eyeline/users/jphilip/test_delighting/koichi_cvpr26/", video_name + ".mp4")
        mode = "lighting"
        #lighting_r_1_frame_00341.exr
        #olats_albedo_frame_00078.exr

    #determine video
    vr = VideoReader(video_path, ctx=cpu(0), width = 832, height = 480) 
    stride = 3
    while stride * (49 - 1) >= len(vr):
        stride = stride  - 1
    assert(stride >= 1)
    idx = np.arange(0, stride*(49-1) + 1, stride)
    
    #enumerate reference images
    tmp = os.listdir(path_folder)
    ref_path = []
    for tt in tmp:
        if mode in tt:
            ref_path.append(os.path.join(path_folder, tt))
    ref_path.sort()

    #load reference images
    frames = []
    for i in idx:
        image, header = read_exr(ref_path[i])
        image = image/(1+image) # simple tonemapping
        # another possible solution: image = Lin_to_Log(image), image = np.clip(image,0,1)
        image = Lin_to_sRGB(image)
        print("Converted image to sRGB color space.")
        image = (image * 255).astype(np.uint8)
        image = cv2.resize(image, (832, 480))
        frames.append(image)
    frames = np.array(frames)
    
    save_video_jordan(tmp_file, frames, fps = 15, quality=9)

    return tmp_file
    
def get_vis_video(path):
    path = os.path.join("/root/", path)
    tmp_folder = "./../tmp_webpage/tmp_vis/" + ((path.split("/")[-1]).split(".mp4")[0]) + "_720"
    os.makedirs(tmp_folder, exist_ok = True)
    if os.path.exists(os.path.join(tmp_folder, "tracking/overlay_tracking.mp4")):
        return os.path.join(tmp_folder, "tracking/overlay_tracking.mp4")
    
    #get track visualization
    lis = path.split("/")
    model_name = lis[-2]
    video_name = lis[-1]
    path = os.path.join("/".join(lis[:-2]), "log/", video_name.split(".mp4")[0], "tmp/track.npy")

    track = np.load(path, allow_pickle = True).item()
    frames = np.zeros((49, track["H"], track["W"], 3))
    vis = Visualizer(
        save_dir=tmp_folder,
        linewidth=2,
        mode="rainbow",
        fps=15,
        tracks_leave_trace=0, #10,   # <= infinite trace
    )

    tr = track["uvz"][None]
    tr = tr[:, :49]
    track["vis"] = track["vis"][None] #(B, T, N, 1)
    track["vis"] = (track["vis"].astype(np.bool_) & (tr[..., 0] >= 0) & (tr[..., 1] >= 0) & (tr[..., 0] < track["W"]) & (tr[..., 1] < track["H"]))

    num_tracks = track["vis"].shape[2]
    
    if False and num_tracks > 250:
        ids = np.random.choice(num_tracks, 250)
        track["vis"] = track["vis"][:, :, ids]
        tr = tr[:, :, ids]
    
    vis.visualize(
        video=torch.from_numpy(frames).permute(0,3,1,2)[None],
        tracks=torch.from_numpy(tr),
        visibility = torch.from_numpy(track["vis"][..., None]),
        filename="overlay",
        save_video=True,
    )
    
    return os.path.join(tmp_folder, "tracking/overlay_tracking.mp4")

def get_ref_imgs(path):
    lis = path.split("/")
    model_name = lis[-2]
    video_name = lis[-1]
    ref_folder = os.path.join("/".join(lis[:-2]), "log/", video_name.split(".mp4")[0], "ref")
    ref_lis = os.listdir(ref_folder)
    ref_lis.sort()
    path_list = []
    idx_list = []
    
    tmp_folder = "./../tmp_webpage/tmp_julien/" + (video_name.split(".mp4"))[0] + "_ref"
    os.makedirs(tmp_folder, exist_ok = True)
    
    for r in ref_lis:
        if not r.endswith(".png"):
            continue
        fr = Image.open(os.path.join(ref_folder, r)).resize((832, 480))
        frames = [fr] * 49
        frames = np.array(frames)
        ref_idx = r.split(".png")[0]        
        tmp_file = os.path.join(tmp_folder, ref_idx + ".png")
        save_video_jordan(tmp_file, frames, fps = 15, quality=9)
        idx_list.append(int(ref_idx))
        path_list.append(tmp_file)
        
    return path_list, idx_list

#classify by video type
dic = {}
for g in good_examples:
    g = g.split("/")
    #vid_name = ((g[-1].split(".mp4"))[0]).replace("_Ref2", "").replace("_Ref1", "")
    #if vid_name not in dic:
    #    dic[vid_name] = []
    #dic[vid_name].append("/".join(g))
    dic[(g[-1].split(".mp4"))[0]] = []
    dic[(g[-1].split(".mp4"))[0]].append("/".join(g))

print("dic", dic)

#Iterate through each video
for video_name in dic:
    examples = dic[video_name]

    if "Original" in video_name:
        continue
    
    #Retrieve ground truth video
    gt_path = get_gt_video(examples[0])
    print("gt", gt_path)
    
    #Retrieve naive video
    naive_path = get_naive_video(examples[0])

    #Retrieve visualization of motion tracks (color gradetion)
    vis_path = get_vis_video(examples[0])
    
    #Stack together (GT, motion tracks, stylized videos)
    """
    Original /  temporal inconsistent video  / ours
    Motiontrack / reference images
    """
    inputs = {
        "Source video": gt_path,
        "Intrinsic Diffusion (per frame)": naive_path,
    }
    for idx, e in enumerate(examples):
        name = "Ours"
        inputs[name] = e
    inputs["Track conditions"] = vis_path

    #attach reference frames
    path_list, idx_list = get_ref_imgs(examples[0])
    for idx, p in enumerate(path_list):
        if idx_list[idx] == 0:
            inputs["First frame"] = p
        elif idx_list[idx] == 48:
            inputs["Last frame"] = p
        else:
            raise NotImplementedError
    
    print("inputs:", inputs)
    
    #Stack videos
    output_video_path = os.path.join("./videos_jul", video_name + ".mp4")
    os.makedirs("./videos_jul",  exist_ok = True)
    stack_videos.stack_videos(inputs, output_video_path, cols=4)
    print("saved:", output_video_path)

