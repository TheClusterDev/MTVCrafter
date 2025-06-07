# motion_extractor.py (Fully Corrected)
import os
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download # Import the downloader
from models.motion4d.vqvae import Encoder, Decoder, VectorQuantizer, SMPL_VQVAE

def get_transforms(w, h, mean, std):
    """Creates a composition of image transformations."""
    return T.Compose([T.Resize((h, w)), T.ToTensor(), T.Normalize(mean, std)])

def get_transforms_from_order(results):
    """Orders YOLO tracking results by frame."""
    order_results = []
    for result in results:
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            for i, id_val in enumerate(result.boxes.id.cpu().numpy()):
                order_results.append({'image': Image.fromarray(result.orig_img[..., ::-1]), 'bbox': boxes[i], 'id': id_val})
    return order_results

def get_transforms_from_mot(order_results):
    """Groups ordered results by track ID."""
    mot_results = {}
    for result in order_results:
        track_id = result['id']
        if track_id not in mot_results: mot_results[track_id] = []
        mot_results[track_id].append(result)
    return list(mot_results.values())

def initialize_motion_models(model_snapshot_path):
    """Initializes and returns all models needed for motion extraction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Initializing motion extraction models...")

    # --- START OF THE CORRECTED FIX ---
    # Explicitly download a working YOLO model from a known-good repository.
    # The original 'ultralytics/yolo_world' does not host 'yolo_world_v2.pt'.
    print("Downloading YOLO World v2 model from a working repository...")
    yolo_model_path = hf_hub_download(
        repo_id="wondervictor/YOLO-World",
        filename="yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth"
    )
    print(f"✅ YOLO model downloaded to {yolo_model_path}")
    
    # Load the YOLO model from the downloaded path
    yolo_model = YOLO(yolo_model_path)
    # --- END OF THE CORRECTED FIX ---

    encoder = Encoder(in_channels=3, mid_channels=[128, 512], out_channels=3072)
    decoder = Decoder(in_channels=3072, mid_channels=[512, 128], out_channels=3)
    vq = VectorQuantizer(nb_code=8192, code_dim=3072, is_train=False)

    vqvae_model = SMPL_VQVAE(encoder, decoder, vq).to(device)
    vqvae_model.load_state_dict(torch.load(os.path.join(model_snapshot_path, "4DMoT/mp_rank_00_model_states.pt"), map_location=device)['module'])
    vqvae_model.eval()

    data_mean = torch.from_numpy(np.load(os.path.join(model_snapshot_path, "dataset_global_mean.npy"))).to(device)
    data_std = torch.from_numpy(np.load(os.path.join(model_snapshot_path, "dataset_global_std.npy"))).to(device)
    
    print("✅ Motion extraction models initialized.")
    return yolo_model, vqvae_model, data_mean, data_std, device

def extract_pkl_from_video(video_path, models_tuple):
    """Extracts motion PKL using the initialized models."""
    yolo_model, vqvae_model, data_mean, data_std, device = models_tuple
    temp_motion_pkl_path = "temp_motion.pkl"
    transforms = get_transforms(512, 512, data_mean, data_std)

    print("Running YOLO object tracking...")
    yolo_results = yolo_model.track(video_path, conf=0.3, iou=0.5, classes=[0], device=device, save=False, show=False, verbose=False)
    
    order_results = get_transforms_from_order(yolo_results)
    mot_results = get_transforms_from_mot(order_results)

    motion_data = {}
    if mot_results:
        # Process only the first tracked person
        tracked_person_frames = mot_results[0]
        image_tensors = torch.stack([transforms(frame['image']) for frame in tracked_person_frames]).to(device)
        
        with torch.no_grad():
            x_encoded = vqvae_model.encoder(image_tensors)
            _, _, vq_output = vqvae_model.vq(x_encoded, return_vq=True)
            motion_data['quant'] = vq_output[1].cpu() 
            motion_data['bbox'] = [frame['bbox'] for frame in tracked_person_frames]

    torch.save(motion_data, temp_motion_pkl_path)
    print(f"Motion data saved to {temp_motion_pkl_path}")
    return temp_motion_pkl_path
