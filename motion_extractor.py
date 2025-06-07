# motion_extractor.py (Corrected with downloader)
import os
import sys
import cv2
import torch
import pickle
from huggingface_hub import hf_hub_download

# --- START OF FIX ---
# Define repository and filename
HF_REPO_ID = "MaNiFest/MaNy"
MODEL_FILENAME = "nlf_l_multi_0.3.2.torchscript"

print(f"Downloading model '{MODEL_FILENAME}' from '{HF_REPO_ID}'...")

# Download the model file from Hugging Face Hub
# This function caches the file, so it will only be downloaded once.
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)

print(f"✅ Model downloaded to: {model_path}")
# --- END OF FIX ---


# Load the TorchScript model once at the top using the downloaded path
assert os.path.exists(model_path), f"Model file not found at {model_path}"
model = torch.jit.load(model_path).cuda().eval()
print("✅ TorchScript model loaded successfully.")

def extract_pkl_from_video(video_path):
    """
    Extracts SMPL pose data from a video file and saves it as a .pkl file.
    
    Args:
        video_path (str): The path to the input video file.

    Returns:
        str: The path to the generated .pkl file.
    """
    output_file = "temp_motion.pkl"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video: {frame_count} frames, {video_width}x{video_height}")

    pose_results = {
        'joints3d_nonparam': [],
    }

    # Use torch.inference_mode() for performance
    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame from BGR (OpenCV default) to RGB and then to a tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).cuda().permute(2, 0, 1).unsqueeze(0)
            
            # Model inference
            pred = model.detect_smpl_batched(frame_tensor)
            
            # Collect pose data, moving it to the CPU to save GPU memory
            if 'joints3d_nonparam' in pred:
                pose_results['joints3d_nonparam'].append(pred['joints3d_nonparam'].cpu())
            else:
                pose_results['joints3d_nonparam'].append(None)

    cap.release()

    # Prepare final output data structure
    output_data = {
        'video_path': video_path,
        'video_length': frame_count,
        'video_width': video_width,
        'video_height': video_height,
        'pose': pose_results
    }

    # Save to a pickle file
    print(f"Saving motion data to '{output_file}'...")
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print("✅ Motion extraction complete.")
    return output_file
