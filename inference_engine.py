# inference_engine.py

import os
import torch
from models.dit.pipeline_mtvcrafter import MTVCrafterPipeline
from PIL import Image

# The model ID for the pre-trained weights on the Hugging Face Hub
MODEL_ID = "yanboding/MTVCrafter"

def run_inference(device, pkl_path, ref_image_path, **kwargs):
    """
    Runs the main video generation inference pipeline.
    """
    # Load the pipeline, which will download weights from the Hub
    # This automatically uses the corrected subfolder logic.
    pipe = MTVCrafterPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    ).to(device)

    # Load reference image and motion data
    ref_image = Image.open(ref_image_path).convert("RGB")
    motion_data = torch.load(pkl_path, map_location=device)
    
    # Generate the video
    video_frames = pipe(
        ref_image,
        motion_data,
        **kwargs
    ).frames[0]
    
    # Define output path and save
    output_video_path = "generated_video.mp4"
    pipe.save_videos(output_video_path, video_frames, fps=25)
    
    return output_video_path
