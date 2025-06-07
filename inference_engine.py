# inference_engine.py (Corrected with the final, verified import path)
import os
import torch
import decord
import imageio
import numpy as np
import pickle
import copy
from PIL import Image
from torchvision.transforms import ToPILImage, transforms, InterpolationMode, functional as F
from huggingface_hub import hf_hub_download

# Import model classes from the cloned repo
from models import MTVCrafterPipeline, Encoder, VectorQuantizer, Decoder, SMPL_VQVAE
from draw_pose import get_pose_images
from utils import concat_images_grid, sample_video, get_sample_indexes, get_new_height_width

# --- START OF FIX ---
# This is the correct, verified import path for the scheduler class
from models.cogvideox_dpm_scheduler import CogVideoXDPMScheduler
# --- END OF FIX ---


def run_inference(device, motion_data_path, ref_image_path='', dst_width=512, dst_height=512, num_inference_steps=50, guidance_scale=3.0, seed=6666):
    num_frames = 49
    to_pil = ToPILImage()
    normalize = transforms.Normalize([0.5], [0.5])
    pipeline_repo_path = "yanboding/MTVCrafter"

    with open(motion_data_path, 'rb') as f:
        data = pickle.load(f)

    # Download pose normalization data
    print("Downloading pose normalization data...")
    mean_path = hf_hub_download(repo_id=pipeline_repo_path, filename="dataset_global_mean.npy")
    std_path = hf_hub_download(repo_id=pipeline_repo_path, filename="dataset_global_std.npy")
    pe_mean = np.load(mean_path)
    pe_std = np.load(std_path)
    print("✅ Pose data loaded.")

    # Manually create the scheduler instance.
    print("Creating DPM Scheduler...")
    scheduler = CogVideoXDPMScheduler.from_config({
        "beta_end": 0.02,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
        "steps_offset": 1,
        "trained_betas": None
    })
    print("✅ Scheduler created.")

    print("Initializing MTVCrafter Pipeline...")
    pipe = MTVCrafterPipeline.from_pretrained(
        model_path=pipeline_repo_path,
        torch_dtype=torch.bfloat16,
        scheduler=scheduler, # Pass the manually created scheduler
    ).to(device)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    print("✅ Pipeline initialized.")

    # Load VQVAE model
    print("Loading VQ-VAE model...")
    vqvae_model_path = hf_hub_download(
        repo_id=pipeline_repo_path,
        filename="4DMoT/mp_rank_00_model_states.pt"
    )
    state_dict = torch.load(vqvae_model_path, map_location="cpu")

    motion_encoder = Encoder(in_channels=3, mid_channels=[128, 512], out_channels=3072, downsample_time=[2, 2], downsample_joint=[1, 1])
    motion_quant = VectorQuantizer(nb_code=8192, code_dim=3072, is_train=False)
    motion_decoder = Decoder(in_channels=3072, mid_channels=[512, 128], out_channels=3, upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0], joint_upsample_rate=[1.0, 1.0])
    vqvae = SMPL_VQVAE(motion_encoder, motion_decoder, motion_quant).to(device)
    vqvae.load_state_dict(state_dict['module'], strict=True)
    print("✅ VQ-VAE model loaded.")

    # (The rest of the script remains the same)
    new_height, new_width = get_new_height_width(data, dst_height, dst_width)
    x1 = (new_width - dst_width) // 2
    y1 = (new_height - dst_height) // 2
    sample_indexes = get_sample_indexes(data['video_length'], num_frames, stride=1)
    ref_image = Image.open(ref_image_path).convert("RGB")
    ref_image = torch.from_numpy(np.array(ref_image)).permute(2, 0, 1).contiguous()
    ref_images = torch.stack([ref_image.clone() for _ in range(num_frames)])
    ref_images = F.resize(ref_images, (new_height, new_width), InterpolationMode.BILINEAR)
    ref_images = F.crop(ref_images, y1, x1, dst_height, dst_width)
    smpl_poses = np.array([pose[0][0].cpu().numpy() for pose in data['pose']['joints3d_nonparam']])
    poses = smpl_poses[sample_indexes]
    norm_poses = torch.tensor((poses - pe_mean) / pe_std)
    offset = [data['video_height'], data['video_width'], 0]
    pose_images_before = get_pose_images(copy.deepcopy(poses), offset)
    pose_images_before = [image.resize((new_width, new_height)).crop((x1, y1, x1+dst_width, y1+dst_height)) for image in pose_images_before]
    input_smpl_joints = norm_poses.unsqueeze(0).to(device)
    with torch.no_grad():
        motion_tokens, _, _ = vqvae(input_smpl_joints, return_vq=True)
        output_motion, _ =  vqvae(input_smpl_joints)
    pose_images_after = get_pose_images(output_motion[0].cpu().detach().numpy() * pe_std + pe_mean, offset)
    pose_images_after = [image.resize((new_width, new_height)).crop((x1, y1, x1+dst_width, y1+dst_height)) for image in pose_images_after]
    ref_images = ref_images / 255.0
    ref_images = normalize(ref_images)

    print("Running main inference...")
    output_images = pipe(
        height=dst_height, width=dst_width, num_frames=num_frames,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, seed=seed,
        ref_images=ref_images.to(torch.bfloat16), motion_embeds=motion_tokens.to(torch.bfloat16),
        joint_mean=pe_mean, joint_std=pe_std,
    ).frames[0]
    print("✅ Inference complete.")

    vis_images = []
    pil_ref_image = to_pil(((ref_images[0] + 1) * 127.5).clamp(0, 255).to(torch.uint8))
    for k in range(len(output_images)):
        vis_image = [pil_ref_image, pose_images_before[k], pose_images_after[k], output_images[k]]
        vis_image = concat_images_grid(vis_image, cols=len(vis_image), pad=2)
        vis_images.append(vis_image)

    output_path = "output.mp4"
    imageio.mimsave(output_path, vis_images, fps=15)
    print(f"✅ Output video saved to {output_path}")
    return output_path
