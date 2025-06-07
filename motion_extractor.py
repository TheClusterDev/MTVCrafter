# motion_extractor.py

import os
import torch
import numpy as np
from ultralytics import YOLO
from models.motion4d.vqvae import VQVAE
from dataset import get_transforms

# Pre-load the models and necessary data
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolo_world_v2.pt")
vqvae_model = VQVAE(in_channel=3, n_embed=8192, embed_dim=256, n_hid=256).to(device)
vqvae_model.load_state_dict(torch.load("4DMoT/mp_rank_00_model_states.pt", map_location=device)['module'])
vqvae_model.eval()

data_mean = torch.from_numpy(np.load("dataset_global_mean.npy")).to(device)
data_std = torch.from_numpy(np.load("dataset_global_std.npy")).to(device)

def extract_pkl_from_video(video_path):
    """
    Extracts motion data from a video file and saves it to a .pkl file.
    This function combines YOLO detection with the VQ-VAE model.
    """
    dst_width = 512
    dst_height = 512
    temp_motion_pkl_path = "temp_motion.pkl"

    transforms = get_transforms(dst_width, dst_height, data_mean, data_std)
    
    # Run YOLO detection
    results = yolo_model.track(
        video_path,
        conf=0.3,
        iou=0.5,
        classes=[0], # Class for 'person'
        device=device,
        save=False,
        show=False,
        verbose=False
    )
    
    with torch.no_grad():
        motion_data = vqvae_model.extract(results, transforms=transforms)
    
    torch.save(motion_data, temp_motion_pkl_path)
    
    return temp_motion_pkl_path
