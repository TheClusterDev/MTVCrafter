# inference_engine.py
import os
import torch
from models.dit.pipeline_mtvcrafter import MTVCrafterPipeline
from PIL import Image

def run_inference(model_path, device, pkl_path, ref_image_path, **kwargs):
    pipe = MTVCrafterPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    ref_image = Image.open(ref_image_path).convert("RGB")
    motion_data = torch.load(pkl_path, map_location=device)
    
    video_frames = pipe(ref_image, motion_data, **kwargs).frames[0]
    
    output_video_path = "generated_video.mp4"
    pipe.save_videos(output_video_path, video_frames, fps=25)
    return output_video_path
