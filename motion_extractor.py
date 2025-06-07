# motion_extractor.py
import os
import sys
import cv2
import torch
import pickle
import torchvision

# Load the TorchScript model once at the top
model_path = 'nlf_l_multi_0.3.2.torchscript'
assert os.path.exists(model_path), f"Model file not found at {model_path}"
model = torch.jit.load(model_path).cuda().eval()

def extract_pkl_from_video(video_path):
    output_file = "temp_motion.pkl"
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_results = {
        'joints3d_nonparam': [],
    }

    with torch.inference_mode(), torch.device('cuda'):
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to tensor
            frame_tensor = torch.from_numpy(frame).cuda()
            frame_batch = frame_tensor.unsqueeze(0).permute(0,3,1,2)
            # Model inference
            pred = model.detect_smpl_batched(frame_batch)
            # Collect pose data
            for key in pose_results.keys():
                if key in pred:
                    #pose_results[key].append(pred[key].cpu().numpy())
                    pose_results[key].append(pred[key])
                else:
                    pose_results[key].append(None)

            frame_idx += 1

    cap.release()

    # Prepare output data
    output_data = {
        'video_path': video_path,
        'video_length': frame_count,
        'video_width': video_width,
        'video_height': video_height,
        'pose': pose_results
    }

    # Save to pkl file
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)

    return output_file
