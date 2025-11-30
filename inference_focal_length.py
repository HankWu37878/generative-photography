# ============================
#  inference_focal_length_img_ddim.py
#  (image → varying focal length with DDIM inversion)
# ============================

import os
import torch
import logging
import argparse
import json
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange

from genphoto.pipelines.pipeline_animation import GenPhotoPipeline
from genphoto.models.unet import UNet3DConditionModelCameraCond
from genphoto.models.camera_adaptor import CameraCameraEncoder, CameraAdaptor
from genphoto.utils.util import save_videos_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# Camera embedding (same as before)
# ===========================

def create_focal_length_embedding(focal_length_values, target_height, target_width,
                                  base_focal_length=24.0, sensor_height=24.0, sensor_width=36.0):
    device = 'cpu'
    focal_length_values = focal_length_values.to(device)
    f = focal_length_values.shape[0]

    sensor_width = torch.tensor(sensor_width, device=device)
    sensor_height = torch.tensor(sensor_height, device=device)
    base_focal_length = torch.tensor(base_focal_length, device=device)

    base_fov_x = 2.0 * torch.atan(sensor_width * 0.5 / base_focal_length)
    base_fov_y = 2.0 * torch.atan(sensor_height * 0.5 / base_focal_length)

    target_fov_x = 2.0 * torch.atan(sensor_width * 0.5 / focal_length_values)
    target_fov_y = 2.0 * torch.atan(sensor_height * 0.5 / focal_length_values)

    crop_ratio_xs = target_fov_x / base_fov_x
    crop_ratio_ys = target_fov_y / base_fov_y

    center_h, center_w = target_height // 2, target_width // 2

    focal_length_embedding = torch.zeros((f, 3, target_height, target_width), dtype=torch.float32)

    for i in range(f):
        crop_h = max(1, min(target_height, torch.round(crop_ratio_ys[i] * target_height).int().item()))
        crop_w = max(1, min(target_width, torch.round(crop_ratio_xs[i] * target_width).int().item()))
        focal_length_embedding[i, :,
                               center_h - crop_h // 2:center_h + crop_h // 2,
                               center_w - crop_w // 2:center_w + crop_w // 2] = 1.0
    return focal_length_embedding

class Camera_Embedding(Dataset):
    def __init__(self, focal_length_values, tokenizer, text_encoder, device, sample_size=[256, 384]):
        self.focal_length_values = focal_length_values.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.sample_size = sample_size

    def load(self):
        f = len(self.focal_length_values)
        prompts = [f"<focal length: {fl.item()}>" for fl in self.focal_length_values]

        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state

        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            diff = encoder_hidden_states[i] - encoder_hidden_states[i - 1]
            differences.append(diff.unsqueeze(0))
        differences.append((encoder_hidden_states[-1] - encoder_hidden_states[0]).unsqueeze(0))

        concatenated_differences = torch.cat(differences, dim=0)
        pad_length = 128 - concatenated_differences.size(1)
        if pad_length > 0:
            concatenated_differences = F.pad(concatenated_differences, (0, 0, 0, pad_length))

        # reshape differences to [B=1, C, F, H, W]
        ccl_embedding = concatenated_differences.unsqueeze(0)  # [1, F, seq_len]
        ccl_embedding = ccl_embedding.unsqueeze(2).unsqueeze(3)  # [1, F, 1, 1, seq_len]
        ccl_embedding = ccl_embedding.expand(-1, -1, 3, self.sample_size[0], self.sample_size[1])  # [1, F, 3, H, W]
        ccl_embedding = ccl_embedding.permute(0, 2, 1, 3, 4)  # [B=1, C=3, F, H, W]

        # focal length embedding: [F, 3, H, W] -> add batch dim
        focal_length_embedding = create_focal_length_embedding(self.focal_length_values, self.sample_size[0], self.sample_size[1]).to(self.device)
        focal_length_embedding = focal_length_embedding.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C=3, F, H, W]

        # concatenate along channel dimension
        camera_embedding = torch.cat([focal_length_embedding, ccl_embedding], dim=1)  # [1, C=6, F, H, W]

        return camera_embedding


# ===========================
# Image loading
# ===========================

def load_img(path, height, width):
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)  # BCHW
    return torch.from_numpy(image)

# ===========================
# Load models
# ===========================

def load_models(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.noise_scheduler_kwargs))
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)
    unet = UNet3DConditionModelCameraCond.from_pretrained_2d(
        cfg.pretrained_model_path, subfolder=cfg.unet_subfolder,
        unet_additional_kwargs=cfg.unet_additional_kwargs
    ).to(device)
    unet.requires_grad_(False)

    camera_encoder = CameraCameraEncoder(**cfg.camera_encoder_kwargs).to(device)
    camera_encoder.requires_grad_(False)
    camera_adaptor = CameraAdaptor(unet, camera_encoder)
    camera_adaptor.requires_grad_(False)
    camera_adaptor.to(device)

    pipeline = GenPhotoPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=noise_scheduler, camera_encoder=camera_encoder
    ).to(device)
    pipeline.enable_vae_slicing()

    return pipeline, device

# ===========================
# Inference with image
# ===========================
def run_inference(pipeline, tokenizer, text_encoder, base_scene, focal_length_list,
                  image_path, output_dir, device, video_length=5, height=256, width=384):
    os.makedirs(output_dir, exist_ok=True)

    # Load image (3 channels)
    image_tensor = load_img(image_path, height, width).to(device)
    
    # --- DDIM inversion ---
    logger.info("Performing DDIM inversion...")
    inverted_latents = pipeline.invert_image_ddim(image=image_tensor)

    # Ensure latents have explicit frame dimension [B, C, F, H, W]
    if inverted_latents.ndim == 4:  # [B, C, H, W]
        inverted_latents = inverted_latents.unsqueeze(2).repeat(1, 1, video_length, 1, 1)
    elif inverted_latents.ndim == 5 and inverted_latents.shape[2] != video_length:
        inverted_latents = inverted_latents.repeat(1, 1, video_length, 1, 1)

    # --- Camera embedding ---
    focal_length_values = torch.tensor(json.loads(focal_length_list), dtype=torch.float32).to(device)

    camera_embedding = Camera_Embedding(focal_length_values, tokenizer, text_encoder, device).load()

    # --- Run pipeline ---
    with torch.no_grad():
        output = pipeline(
            prompt=base_scene,
            camera_embedding=camera_embedding,
            video_length=video_length,
            height=height,
            width=width,
            guidance_scale=8.0,
            num_inference_steps=25,
            latents=inverted_latents
        )

    sample = output.videos[0]  # get first video

    # Save output
    save_path = os.path.join(output_dir, "sample.gif")
    save_videos_grid(sample[None, ...], save_path)
    logger.info(f"Saved generated sample to {save_path}")





# ===========================
# Main
# ===========================
def main(config_path, base_scene, focal_length_list, image_path):
    torch.manual_seed(42)
    cfg = OmegaConf.load(config_path)
    pipeline, device = load_models(cfg)

    run_inference(
        pipeline=pipeline,
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        base_scene=base_scene,
        focal_length_list=focal_length_list,
        image_path=image_path,
        output_dir=cfg.output_dir,
        device=device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_scene", type=str, required=True)
    parser.add_argument("--focal_length_list", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.base_scene, args.focal_length_list, args.image_path)
