"""
1. Run `stack_videos.py` to stack all your comparison videos into a 2x3 or 2x4 video grids with titles. 

2. Edit `make_webpage.py` to replace text content and correct paths for each video sample. 

3. Run `make_webpage.py` to generate the html file `supplemental.html`
python make_webpage.py

4. zip
cd ./../
zip -r webpage.zip webpage
"""

import os

def generate_html(data, output_filename="index.html"):
    """
    Generates a sticky-nav supplementary HTML file from a data dictionary.
    """
    
    # --- 1. CSS & HEAD (The styling we perfected) ---
    css = """
    <style>
        html { scroll-padding-top: 100px; }
        body {
            font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 14pt;
            line-height: 1.6;
            color: #333;
            background-color: #ffffff;
            margin: 0;
            padding: 0;
        }
        .container {
            width: 98%;
            max-width: 2400px; 
            margin: 0 auto;
            padding: 20px;
        }
        video {
            width: 100%; 
            height: auto;
            display: block;
            margin: 32px auto;
            border-radius: 4px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            background: #000;
        }
        .video-row {
            display: flex;
            flex-wrap: wrap;
            width: 100%;
            gap: 16px;
            margin: 32px 0px;
            justify-content: center;
        }
        .ours-720p {
            flex: 1;
            min-width: 45%;
            margin: 0;
        }
        .ab-tp { width: 100%; max-width: 100%; margin: 0 auto; }
        h1, h2, h3 { text-align: center; color: #111; }
        h1 { margin: 40px 0 12px 0; }
        h2 { 
            margin: 80px 0 24px 0; 
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        h3 { margin: 50px 0 20px 0; }
        p { text-align: justify; max-width: 1200px; margin-left: auto; margin-right: auto; }
        ol { max-width: 1200px; margin-left: auto; margin-right: auto; }
        .exp {
            margin: 8px 0px 48px 0px;
            font-style: italic;
            text-align: center;
            color: #555;
            font-size: 1.1rem;
        }
        #toc {
            position: sticky;
            top: 0; left: 0; width: 100%;
            background-color: #1a202c; color: white;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            padding: 14px 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #toc ul {
            list-style-type: none; padding: 0; margin: 0;
            display: flex; flex-wrap: wrap; justify-content: center; gap: 12px;
        }
        #toc li { margin: 0; }
        #toc a {
            text-decoration: none; color: #fff;
            font-size: 1.1rem; font-weight: 600;
            padding: 8px 18px;
            background-color: rgba(255,255,255,0.1);
            border-radius: 6px; transition: all 0.2s ease;
        }
        #toc a:hover { background-color: #4a90e2; transform: translateY(-1px); }
        #toc li.active a {
            background-color: #4a90e2; color: white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }
    </style>
    """

    javascript = """
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            var tocContainer = document.getElementById("toc");
            var tocList = document.createElement("ul");
            
            // Select ONLY h3s for the nav
            var headings = document.querySelectorAll("h3");
            
            headings.forEach(function(heading) {
                if (!heading.id) {
                    heading.id = heading.textContent.toLowerCase().trim()
                        .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
                }
                var li = document.createElement("li");
                var a = document.createElement("a");
                a.href = "#" + heading.id;
                a.textContent = heading.textContent;
                li.appendChild(a);
                tocList.appendChild(li);
            });
            tocContainer.appendChild(tocList);

            // Scroll Spy
            window.addEventListener("scroll", function() {
                var scrollPos = window.scrollY || document.documentElement.scrollTop;
                var currentId = "";
                headings.forEach(function(heading) {
                    if (heading.offsetTop <= (scrollPos + 120)) { currentId = heading.id; }
                });
                document.querySelectorAll('#toc li').forEach(function(li) {
                    li.classList.remove('active');
                });
                if (currentId) {
                    var activeLink = document.querySelector('#toc a[href="#' + currentId + '"]');
                    if (activeLink) activeLink.parentElement.classList.add('active');
                }
            });

            var videos = document.querySelectorAll("video");
            for (var i = 0; i < videos.length; i++) {
                videos[i].autoplay = true; videos[i].loop = true;
                videos[i].muted = true; videos[i].setAttribute("playsinline", ""); 
            }
        });
    </script>
    """

    # --- 2. CONTENT BUILDER ---
    html_content = []
    html_content.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data.get('title', 'Supplementary Material')}</title>
    {css}
</head>
<body>
    <nav id="toc"></nav>
    <div class="container">
        <h1 style="margin-bottom: 12px">{data.get('title', 'Project Title')}</h1>
        <p style="text-align: center; font-size: x-large; margin-top: 12px;">{data.get('subtitle', 'Supplementary Material')}</p>
        <p>{data.get('abstract', '')}</p>
    """)

    # Loop through blocks
    for block in data.get('blocks', []):
        b_type = block.get('type')
        content = block.get('content')
        
        if b_type == 'h2':
            html_content.append(f'<h2 id="{block.get("id", "")}">{content}</h2>')
        
        elif b_type == 'h3':
            html_content.append(f'<h3 id="{block.get("id", "")}">{content}</h3>')
        
        elif b_type == 'text':
            html_content.append(f'<p>{content}</p>')
        
        elif b_type == 'video':
            # content should be path string
            html_content.append(f'<video><source src="{content}" type="video/mp4" /></video>')
            
        elif b_type == 'video_row':
            # content should be list of paths
            html_content.append('<div class="video-row">')
            for vid_path in content:
                html_content.append(f'<video class="ours-720p"><source src="{vid_path}" type="video/mp4" /></video>')
            html_content.append('</div>')

        elif b_type == 'captioned_video':
            # content should be dict: {'path': str, 'caption': str}
            # 'caption' can contain HTML like <b>
            v_path = content.get('path')
            caption = content.get('caption')
            html_content.append(f'<video><source src="{v_path}" type="video/mp4" /></video>')
            html_content.append(f'<p class="exp" style="text-align: center;">{caption}</p>')

        elif b_type == 'list':
            # content: {'style': '1' or 'I', 'items': []}
            list_style = content.get('style', '1')
            items = content.get('items', [])
            html_content.append(f'<ol type="{list_style}">')
            for item in items:
                html_content.append(f'<li>{item}</li>')
            html_content.append('</ol>')

    # Close container and add scripts
    html_content.append("</div>")
    html_content.append(javascript)
    html_content.append("</body></html>")

    # Write file
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))
    
    print(f"Successfully generated {output_filename}")


# --- 3. THE INPUT DATA (Dictionary of Keys) ---
# You can edit this dictionary to change the website content easily.

#Prepare baselines
topk = 30

baselines = []
for f in os.listdir("./baselines")[:topk]:
    path = os.path.join("./baselines", f)
    baselines.append({"type": "video", "content": path})
    
videos_lady = []
for f in ["recon.mp4", "remove.mp4", "rep.mp4"]:
    path = os.path.join("./lady", f)
    videos_lady.append({"type": "video", "content": path})

#Prepare Motion control
videos_mo = []
for f in os.listdir("./videos_motion_transfer")[:topk]:
    path = os.path.join("./videos_motion_transfer", f)
    videos_mo.append({"type": "video", "content": path})

#Prepare Motion control 720P
#videos_mo_720 = []
#for f in os.listdir("./720P_motion_transfer")[:topk]:
#    path = os.path.join("./720P_motion_transfer", f)
#    videos_mo_720.append({"type": "video", "content": path})

#Prepare Mesh based stylization
videos_mesh = []
for f in os.listdir("./videos_mesh")[:topk]:
    path = os.path.join("./videos_mesh", f)
    videos_mesh.append({"type": "video", "content": path})

#Prepare camera_First
videos_camera_first = []
for f in os.listdir("./camera_First")[:topk]:
    path = os.path.join("./camera_First", f)
    videos_camera_first.append({"type": "video", "content": path})
for f in os.listdir("./camera_custom_sig")[:topk]:
    path = os.path.join("./camera_custom_sig", f)
    videos_camera_first.append({"type": "video", "content": path})
for f in os.listdir("./static_z")[:topk]:
    path = os.path.join("./static_z", f)
    videos_camera_first.append({"type": "video", "content": path})

#Prepare camera_dynamic
videos_camera_dynamic = []
for f in os.listdir("./camera_dynamic")[:topk]:
    path = os.path.join("./camera_dynamic", f)
    videos_camera_dynamic.append({"type": "video", "content": path})
for f in os.listdir("./davis_dynamic")[:topk]:
    path = os.path.join("./davis_dynamic", f)
    videos_camera_dynamic.append({"type": "video", "content": path})


#Prepare Julien
videos_jul = []
lisd = os.listdir("./videos_jul")
lisd.sort()
for f in lisd[:topk]:
    path = os.path.join("./videos_jul", f)
    videos_jul.append({"type": "video", "content": path})

#Prepare sample
videos_sample = []
lisd = os.listdir("./sampling_videos")
lisd.sort()
for f in lisd[:topk]:
    path = os.path.join("./sampling_videos", f)
    videos_sample.append({"type": "video", "content": path})

content_data = {
    "title": "Go-with-the-Track: Video Compositing and Motion Control with Point Tracking",

    "subtitle": "Anonymous SIGGRAPH submission &nbsp;|&nbsp; Paper ID 483",

    "abstract": "Go-with-the-Track unifies motion control and multi-reference conditioning within a single framework, using dense point tracks to guide spatiotemporal evolution. Our model enables the precise placement of elements that enter or exit the scene and achieves superior fidelity in applications ranging from complex motion transfer to dynamic camera control.",

    "blocks": [
        # --- SECTION 3: APPLICATIONS ---
        {"type": "h2", "content": "Applications"},
        {"type": "text", "content": "We show samples of some applications of Go-with-the-Track: Motion Transfer, Mesh-based Stylization, Static Scene (Bullet Time) Camera Control, Dynamic-scene Camera Control, Temporal Stabilization"},
        
        {"type": "h3", "content": "Keyframe based Motion Transfer"},
        {"type": "text", "content": "We extract point-track conditions from source videos to transfer complex motion patterns to new subjects defined by reference images. This process preserves the visual identity of the reference while faithfully reproducing the target dynamics."},
        *videos_mo,

        {"type": "h3", "content": "Track-level object editing"},
        {"type": "text", "content": "Our flexible problem formulation allows track-level object editing such as recompositing, removal, and replacement as follows."},
        # 1. Recomposition
        {"type": "text", "content": "<b>Recomposition of human and background:</b>"},
        {"type": "text", "content": "By separating foreground human from background, our model allows to preserve both human and background appearence and motions with just three reference iamges"},
        videos_lady[0],
        # 2. Removal
        {"type": "text", "content": "<b>Removal of human</b>"},
        {"type": "text", "content": "By removing point-tracks and reference image associated with human, we can obtain a video of background."},
        videos_lady[1],
        # 3. Replacement
        {"type": "text", "content": "<b>Appearence replacement</b>"},
        {"type": "text", "content": "By matching keypoints of human body between original human and reference human, we can replace the appearence of human in the video."},
        videos_lady[2],

        # --- SECTION 2: 720P ---
        #{"type": "h3", "content": "Results in 720p"},
        #{"type": "text", "content": "By fine-tuning our 14B parameter model on 65K high-quality real-world videos, we successfully scaled our generation capabilities to 720P. These results demonstrate that TrackR2V maintains precise motion controllability and high visual fidelity even at significantly higher resolutions."},
        #*videos_mo_720,
        
        # --- SECTION 1: VIDEO RESULTS ---
        # {"type": "h2", "content": "Baseline Comparisons"},

        {"type": "h3", "content": "Mesh-based Stylization"},
        {"type": "text", "content": "By combining stylized keyframes with point tracks derived from mesh vertices, our model can render 3D animations in novel artistic styles. This ensures the generated video strictly adheres to the underlying geometry and animation of the original mesh while applying consistent visual themes."},
        *videos_mesh,
        
        {"type": "h3", "content": "Static Scene (Bullet Time) Camera Control"},
        {"type": "text", "content": "Go-with-the-Track enables precise camera retargeting and novel view synthesis by projecting reconstructed points onto target trajectories. We support complex camera movements, including spirals and smooth view interpolations, using just sparse image inputs and point cloud reconstructions. Beyond static environments, our framework handles camera retargeting within dynamic scenes containing independently moving objects. By leveraging robust point-track conditions, we maintain temporal coherence between the shifting camera perspective and the dynamic elements of the scene."},
        *videos_camera_first,

        {"type": "h3", "content": "Dynamic Scene Camera Control"},
        {"type": "text", "content": "Beyond static scene (Bullet Time) video generation, our framework handles camera retargeting within dynamic scenes containing independently moving objects. By leveraging robust point-track conditions, we maintain temporal coherence between the shifting camera perspective and the dynamic elements of the scene."},
        *videos_camera_dynamic,

        {"type": "h3", "content": "Temporal Stabilization"},
        {"type": "text", "content": "We address temporal flickering in inverse rendering tasks by propagating albedo and shading estimates from keyframes to the full sequence. TrackR2V ensures smooth, temporally consistent material properties across frames, effectively smoothing out the jitter often seen in per-frame estimation methods."},
        *videos_jul,

        {"type": "h3", "content": "Comparison to baselines"},
        {"type": "text", "content": "Compared to state-of-the-art methods like ATI, DiffusionAsShader, and Tora, TrackR2V demonstrates superior structural integrity and motion following. Our model significantly reduces artifacts and better preserves subject identity, particularly in challenging scenarios involving occlusions, as validated by user studies."},
        # Since these are just single comparison videos stacked, we add them as 'video' types
        *baselines,

        {"type": "h3", "content": "Point Densification"},
        {"type": "text", "content": "We provide a visuals of the detected point tracks when using our iterative resampling technique (Algorithm 1) versus uniformly sampling of point queries in the space-time regions with point track queries. As evident, we see that our point tracks show much better coverage and less sparsity especially for objects that enter or exit in the middle of the scene."},
        *videos_sample,
    ]
}

if __name__ == "__main__":
    generate_html(content_data, "supplemental.html")