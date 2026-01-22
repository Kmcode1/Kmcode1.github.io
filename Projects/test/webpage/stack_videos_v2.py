import cv2
import imageio.v3 as iio  # For reading
import imageio            # For writing
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def crop_and_resize(img, target_w, target_h):
    """
    Center-crops image to target aspect ratio, then resizes to target dimensions.
    """
    if img is None or img.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    target_ar = target_w / target_h
    current_ar = w / h

    if current_ar > target_ar:
        # Too wide: Crop width
        new_w = int(h * target_ar)
        start_x = (w - new_w) // 2
        img_cropped = img[:, start_x:start_x+new_w]
    else:
        # Too tall: Crop height
        new_h = int(w / target_ar)
        start_y = (h - new_h) // 2
        img_cropped = img[start_y:start_y+new_h, :]

    try:
        img_resized = cv2.resize(img_cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    except cv2.error:
        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
    return img_resized

def add_header(img, text):
    """
    Adds a white header with centered black text using PIL.
    """
    h, w = img.shape[:2]
    header_h = 60 
    font_size = 36

    # Create white background for header
    header_bg = Image.new('RGB', (w, header_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(header_bg)

    try:
        font_names = ["arial.ttf", "Arial.ttf", "times.ttf", "Times New Roman.ttf", "DejaVuSans.ttf"]
        font = None
        for name in font_names:
            try:
                font = ImageFont.truetype(name, font_size)
                break
            except IOError:
                continue
        if font is None:
            raise IOError("No specific fonts found")
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    x = (w - text_w) / 2
    y = (header_h - text_h) / 2 - 4 

    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    header_np = np.array(header_bg)

    return np.vstack((header_np, img))

def load_frames(path, max_frames):
    try:
        data = iio.imread(path, index=None)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return [np.zeros((100, 100, 3), dtype=np.uint8)] # Placeholder

    raw_frames = []
    if data.ndim == 3:
        raw_frames = [data]
    elif data.ndim == 4:
        raw_frames = list(data)
    elif data.ndim == 2:
        bgr = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        raw_frames = [bgr]
    else:
        return [np.zeros((100, 100, 3), dtype=np.uint8)]

    output_frames = []
    source_len = len(raw_frames)
    
    for i in range(max_frames):
        frame = raw_frames[i % source_len]
        output_frames.append(frame)
        
    return output_frames

# --- ORIGINAL GRID FUNCTION ---
def stack_videos_grid(data_dict, output_filename="grid_output.mp4", cols=4):
    TARGET_W = 832
    TARGET_H = 480
    TARGET_FPS = 15
    MAX_FRAMES = 49
    
    print(f"Processing Grid Layout: {len(data_dict)} inputs...")
    
    processed_clips = []
    
    for name, path in data_dict.items():
        raw_sequence = load_frames(path, MAX_FRAMES)
        processed_stream = []
        for frame in raw_sequence:
            res = crop_and_resize(frame, TARGET_W, TARGET_H)
            res = add_header(res, name)
            processed_stream.append(res)
        processed_clips.append(processed_stream)

    try:
        writer = imageio.get_writer(output_filename, fps=TARGET_FPS, codec="libx264", pixelformat="yuv420p")
    except Exception:
        writer = imageio.get_writer(output_filename, fps=TARGET_FPS)

    if processed_clips:
        block_h, block_w = processed_clips[0][0].shape[:2]
    else:
        block_h, block_w = 480+60, 832
    blank_block = np.full((block_h, block_w, 3), 0, dtype=np.uint8)

    print("Stitching frames...")
    for f_idx in range(MAX_FRAMES):
        current_frame_cells = []
        for clip in processed_clips:
            current_frame_cells.append(clip[f_idx])
        while len(current_frame_cells) % cols != 0:
            current_frame_cells.append(blank_block)
        
        grid_rows = []
        for i in range(0, len(current_frame_cells), cols):
            row_imgs = current_frame_cells[i:i+cols]
            grid_rows.append(np.hstack(row_imgs))
            
        final_frame = np.vstack(grid_rows)
        writer.append_data(final_frame)

    writer.close()
    print(f"Success! Saved to {output_filename}")

# --- NEW SPECIAL LAYOUT FUNCTION ---
def stack_special_layout(data_dict, output_filename="special_output.mp4"):
    """
    Layout:
    [ A (Motion Track) ] [ B (Ground Truth) ]
    [           C (Generated 720p)          ]
    
    Constraint: Width(A) + Width(B) = Width(C)
    """
    TARGET_FPS = 15
    MAX_FRAMES = 49
    
    # Target Dimensions
    # C is standard 720p
    C_W, C_H = 1280, 720
    # A and B share the width of C, so divide by 2
    AB_W = C_W // 2  # 640
    AB_H = 360       # 16:9 aspect ratio relative to 640 width

    print("Processing Special Layout (A+B over C)...")

    # 1. Pre-load and Process A
    print(" - Processing Video A...")
    frames_A = load_frames(data_dict['A']['path'], MAX_FRAMES)
    proc_A = []
    for f in frames_A:
        res = crop_and_resize(f, AB_W, AB_H)
        res = add_header(res, data_dict['A']['name'])
        proc_A.append(res)

    # 2. Pre-load and Process B
    print(" - Processing Video B...")
    frames_B = load_frames(data_dict['B']['path'], MAX_FRAMES)
    proc_B = []
    for f in frames_B:
        res = crop_and_resize(f, AB_W, AB_H)
        res = add_header(res, data_dict['B']['name'])
        proc_B.append(res)

    # 3. Pre-load and Process C
    print(" - Processing Video C...")
    frames_C = load_frames(data_dict['C']['path'], MAX_FRAMES)
    proc_C = []
    for f in frames_C:
        res = crop_and_resize(f, C_W, C_H)
        res = add_header(res, data_dict['C']['name'])
        proc_C.append(res)

    # 4. Write Video
    try:
        writer = imageio.get_writer(output_filename, fps=TARGET_FPS, codec="libx264", pixelformat="yuv420p")
    except Exception:
        writer = imageio.get_writer(output_filename, fps=TARGET_FPS)

    print("Stitching frames...")
    for i in range(MAX_FRAMES):
        # Row 1: A + B
        row1 = np.hstack((proc_A[i], proc_B[i]))
        
        # Row 2: C
        row2 = proc_C[i]
        
        # Stack Rows
        final_frame = np.vstack((row1, row2))
        
        writer.append_data(final_frame)

    writer.close()
    print(f"Success! Saved to {output_filename}")


if __name__ == "__main__":
    
    # --- MODE 1: Standard Grid ---
    # grid_inputs = { ... }
    # stack_videos_grid(grid_inputs, "grid.mp4", cols=4)

    # --- MODE 2: Special Layout (A+B / C) ---
    special_inputs = {
        'A': {'name': 'Motion Track', 'path': 'raw_videos/recam.mp4'},
        'B': {'name': 'Ground Truth', 'path': 'raw_videos/source.mp4'},
        'C': {'name': 'Vista4D (Ours)', 'path': 'raw_videos/ours.mp4'}
    }

    try:
        stack_special_layout(special_inputs, "output_special_layout.mp4")
    except Exception as e:
        print(f"An error occurred: {e}")