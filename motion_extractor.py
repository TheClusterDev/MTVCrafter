# motion_extractor.py

import os
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

# Correctly import the classes available in your vqvae.py file
from models.motion4d.vqvae import Encoder, Decoder, VectorQuantizer, SMPL_VQVAE

# --- Helper Functions (to process YOLO output) ---

def get_transforms(w, h, mean, std):
    """Defines the standard image transformations for the model."""
    return T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

def get_transforms_from_order(results):
    """Processes raw YOLO results into an ordered list of detections."""
    order_results = []
    for result in results:
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            for i, id_val in enumerate(result.boxes.id.cpu().numpy()):
                order_result = {
                    'image': Image.fromarray(result.orig_img[..., ::-1]),
                    'bbox': boxes[i],
                    'id': id_val
                }
                order_results.append(order_result)
    return order_results

def get_transforms_from_mot(order_results):
    """Groups ordered detections by their tracking ID."""
    mot_results = {}
    for result in order_results:
        track_id = result['id']
        if track_id not in mot_results:
            mot_results[track_id] = []
        mot_results[track_id].append(result)
    return list(mot_results.values())

# --- Main Extractor Logic ---

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading motion extraction models...")
yolo_model = YOLO("yolo_world_v2.pt")

# Initialize the components required by SMPL_VQVAE
encoder = Encoder(in_channels=3, mid_channels=[128, 512], out_channels=3072)
decoder = Decoder(in_channels=3072, mid_channels=[512, 128], out_channels=3)
vq = VectorQuantizer(nb_code=8192, code_dim=3072, is_train=False)

# Create the main model and load its weights
vqvae_model = SMPL_VQVAE(encoder, decoder, vq).to(device)
vqvae_model.load_state_dict(torch.load("4DMoT/mp_rank_00_model_states.pt", map_location=device)['module'])
vqvae_model.eval()

data_mean = torch.from_numpy(np.load("dataset_global_mean.npy")).to(device)
data_std = torch.from_numpy(np.load("dataset_global_std.npy")).to(device)
print("✅ Motion extraction models loaded.")


def extract_pkl_from_video(video_path):
    """
    Extracts motion data from a video file by running YOLO and the VQ-VAE encoder.
    """
    dst_width, dst_height = 512, 512
    temp_motion_pkl_path = "temp_motion.pkl"
    transforms = get_transforms(dst_width, dst_height, data_mean, data_std)

    print("Running YOLO object tracking on video...")
    yolo_results = yolo_model.track(
        video_path, conf=0.3, iou=0.5, classes=[0], device=device, save=False, show=False, verbose=False
    )
    print("YOLO tracking complete. Processing results...")

    order_results = get_transforms_from_order(yolo_results)
    mot_results = get_transforms_from_mot(order_results)

    motion_data = {}
    if mot_results:
        # Use the first tracked person's motion
        tracked_person_frames = mot_results[0]
        
        # Stack all frame images into a single tensor for batch processing
        image_tensors = torch.cat([transforms(frame['image']).unsqueeze(0) for frame in tracked_person_frames]).to(device)
        
        with torch.no_grad():
            # The SMPL_VQVAE forward pass will encode the whole batch
            # Note: The forward pass returns a tuple, we need the encoded representation.
            # We assume the model's forward pass can handle this and we need to call encode.
            # Based on the vqvae.py code, the model doesn't have a simple .encode method.
            # We will call the encoder and vq components directly.
            x = image_tensors.permute(0, 3, 1, 2) # Reshape to (B, C, H, W)
            x_encoded = vqvae_model.encoder(x)
            _, quant_loss, vq_output = vqvae_model.vq(x_encoded, return_vq=True)
            
            # The vq_output should contain the quantized vectors or indices
            # For this model, it returns the flattened vectors and the loss.
            # We save the quantized vectors themselves.
            motion_data['quant'] = vq_output.cpu() 
            motion_data['bbox'] = [frame['bbox'] for frame in tracked_person_frames]

    torch.save(motion_data, temp_motion_pkl_path)
    print(f"Motion data saved to {temp_motion_pkl_path}")
    return temp_motion_pkl_path
