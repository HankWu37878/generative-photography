import inspect
import torch

import numpy as np
import torchvision.transforms as T
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
from diffusers.utils import is_accelerate_available
from packaging import version
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import deprecate, logging, BaseOutput
import PIL.Image as Image

from genphoto.models.camera_adaptor import CameraCameraEncoder
from genphoto.models.unet import UNet3DConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline, LoraLoaderMixin):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,

        multidiff_total_steps: int = 1,
        multidiff_overlaps: int = 12,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        single_model_length = video_length
        video_length = multidiff_total_steps * (video_length - multidiff_overlaps) + multidiff_overlaps
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred_full = torch.zeros_like(latents).to(latents.device)
                mask_full = torch.zeros_like(latents).to(latents.device)
                noise_preds = []

                for multidiff_step in range(multidiff_total_steps):
                    start_idx = multidiff_step * (single_model_length - multidiff_overlaps)
                    latent_partial = latents[:, :, start_idx: start_idx + single_model_length].contiguous()
                    mask_full[:, :, start_idx: start_idx + single_model_length] += 1

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latent_partial] * 2) if do_classifier_free_guidance else latent_partial
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.to(dtype=latents_dtype)

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_preds.append(noise_pred)

                for pred_idx, noise_pred in enumerate(noise_preds):
                    start_idx = pred_idx * (single_model_length - multidiff_overlaps)
                    noise_pred_full[:, :, start_idx: start_idx + single_model_length] += noise_pred / mask_full[:, :, start_idx: start_idx + single_model_length]

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred_full, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)


class GenPhotoPipeline(AnimationPipeline):
    _optional_components = []

    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet3DConditionModel,
                 scheduler: Union[
                     DDIMScheduler,
                     PNDMScheduler,
                     LMSDiscreteScheduler,
                     EulerDiscreteScheduler,
                     EulerAncestralDiscreteScheduler,
                     DPMSolverMultistepScheduler],
                 camera_encoder: CameraCameraEncoder):

        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)
        self.register_modules(camera_encoder=camera_encoder)

    # ==================== ADD THIS NEW METHOD ====================
    @torch.no_grad()
    def invert_latents_from_image(
        self,
        image_path: str,
        video_length: int,
        height: int,
        width: int,
        text_embeddings: torch.FloatTensor,
        camera_embedding_features: List[torch.FloatTensor],
        device: str = "cuda",
        guidance_scale: float = 7.5,
        num_inversion_steps: int = 50,
    ):
        """
        Perform DDIM inversion from an input image.
        Returns inverted latents at maximum noise level.
        """
        from PIL import Image
        import torchvision.transforms as T
        import sys
        
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"DDIM Inversion Started with Progressive Zoom", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)
        
        # Load and preprocess image
        print(f"Loading original image: {image_path}", file=sys.stderr, flush=True)
        original_image = Image.open(image_path).convert("RGB")
        
        # --- NEW LOGIC: Progressive Center Crop/Zoom ---
        
        # 1. Define the target crop ratios for a zoom-in effect
        # We'll transition from a wide shot (crop_ratio=1.0) to a tight shot (crop_ratio=0.4)
        # using a linear interpolation over the video_length frames.
        # This simulates focal lengths 25mm -> 65mm.
        
        # Start ratio (e.g., 25mm) and End ratio (e.g., 65mm)
        start_ratio = 1.0
        end_ratio = 0.4 
        
        # Create a linear progression of ratios
        crop_ratios = [
            start_ratio - (start_ratio - end_ratio) * (i / (video_length - 1))
            for i in range(video_length)
        ]
        
        print(f"Zoom Ratios (Start: {crop_ratios[0]:.2f}, End: {crop_ratios[-1]:.2f})", file=sys.stderr, flush=True)
        
        # Define the VAE preprocessing transform (Resize and Normalize)
        # Note: We apply the crop *manually* before this transform.
        vae_transform = T.Compose([
            T.Resize((height, width)), # Resize the cropped image back to target HxW
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        
        all_latents = []
        
        for i in range(video_length):
            crop_ratio = crop_ratios[i]
            orig_w, orig_h = original_image.size
            
            # Calculate the crop dimensions
            new_w = int(orig_w * crop_ratio)
            new_h = int(orig_h * crop_ratio)
            
            # Calculate the top-left corner for a center crop
            left = (orig_w - new_w) // 2
            top = (orig_h - new_h) // 2
            right = left + new_w
            bottom = top + new_h
            
            # Perform the center crop
            zoomed_image = original_image.crop((left, top, right, bottom))
            
            # Apply VAE preprocessing (Resize back to HxW and Normalize)
            img_tensor = vae_transform(zoomed_image).unsqueeze(0).to(device=device, dtype=text_embeddings.dtype)
            
            # Encode to latent space
            with torch.no_grad():
                latent = self.vae.encode(img_tensor).latent_dist.sample()
                latent = latent * 0.18215  # VAE scaling factor
                
            all_latents.append(latent)
            
            if i == 0 or i == video_length // 2 or i == video_length - 1:
                print(f"  Frame {i+1}/{video_length} | Crop Ratio: {crop_ratio:.2f} | Latent shape: {latent.shape}", file=sys.stderr, flush=True)

        # Combine all latents into the final 5D tensor (B, C, F, H, W)
        latent_5d = torch.stack(all_latents, dim=2)
        
        print(f"Combined 5D latent shape: {latent_5d.shape}", file=sys.stderr, flush=True)
        print(f"Latent stats - min: {latent_5d.min().item():.4f}, max: {latent_5d.max().item():.4f}, mean: {latent_5d.mean().item():.4f}", file=sys.stderr, flush=True)
        
        # --- END NEW LOGIC ---
        
        # DDIM inversion - go from x_0 to x_T
        self.scheduler.set_timesteps(num_inversion_steps, device=device)
        timesteps = list(reversed(self.scheduler.timesteps))  # Reverse for inversion
        print(f"Inversion timesteps: {len(timesteps)}", file=sys.stderr, flush=True)
        print(f"First timestep: {timesteps[0]}, Last timestep: {timesteps[-1]}", file=sys.stderr, flush=True)
        
        x = latent_5d.clone()
        do_cfg = guidance_scale > 1.0
        
        for i, t in enumerate(timesteps):
            if i % 10 == 0:
                print(f"  Inversion step {i}/{len(timesteps)}, t={t.item()}", file=sys.stderr, flush=True)
                print(f"    Latent stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}", file=sys.stderr, flush=True)
            
            # Prepare input for CFG
            if do_cfg:
                latent_model_input = torch.cat([x] * 2)
            else:
                latent_model_input = x
            
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # UNet prediction with camera features
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    camera_embedding_features=camera_embedding_features
                ).sample
            
            # Apply CFG
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # DDIM inversion step
            current_t_val = t.item()
            alpha_t = self.scheduler.alphas_cumprod[current_t_val]
            
            if i + 1 < len(timesteps):
                next_t_val = timesteps[i + 1].item()
            else:
                next_t_val = self.scheduler.timesteps[0].item()
            
            alpha_t_next = self.scheduler.alphas_cumprod[next_t_val]
            
            # Inversion formula: go from less noisy to more noisy
            predicted_x0 = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            predicted_x0 = predicted_x0.detach()
            
            # Direction to next (noisier) latent
            x =  alpha_t_next.sqrt() * predicted_x0 + (1 - alpha_t_next).sqrt() * noise_pred
            del predicted_x0, noise_pred
            if i % 10 == 0:
                torch.cuda.empty_cache()

            if i >= 35:
              return x
        
        print(f"✓ Inversion complete: {x.shape}", file=sys.stderr, flush=True)
        print(f"Final inverted latent stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}", file=sys.stderr, flush=True)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)
        # x = x * self.scheduler.init_noise_sigma
        return x
    # ==================== END NEW METHOD ====================

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.cpu().float().numpy()
        return video

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        # ... keep original implementation unchanged ...
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        camera_embedding: torch.FloatTensor,
        inversion_camera_embedding: torch.FloatTensor,
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        multidiff_total_steps: int = 1,
        multidiff_overlaps: int = 12,
        # ==================== ADD THESE PARAMETERS ====================
        input_image_path: Optional[str] = None,
        use_inversion: bool = False,
        num_inversion_steps: int = 100,
        # ==================== END NEW PARAMETERS ====================
        **kwargs,
    ):
        import sys
        
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"GenPhotoPipeline __call__", file=sys.stderr, flush=True)
        print(f"use_inversion: {use_inversion}", file=sys.stderr, flush=True)
        print(f"input_image_path: {input_image_path}", file=sys.stderr, flush=True)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)
        
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(prompt, height, width, callback_steps)

        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = camera_embedding[0].device if isinstance(camera_embedding, list) else camera_embedding.device
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        inversion_text_embeddings = self._encode_prompt(
            "", device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )


        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Video length calculations
        single_model_length = video_length
        orig_video_length = video_length
        video_length = multidiff_total_steps * (video_length - multidiff_overlaps) + multidiff_overlaps
        num_channels_latents = self.unet.in_channels

        # ==================== ENCODE CAMERA FEATURES FIRST ====================
        print("Encoding camera embeddings...", file=sys.stderr, flush=True)
        if isinstance(camera_embedding, list):
            assert all([x.ndim == 5 for x in camera_embedding])
            bs = camera_embedding[0].shape[0]
            camera_embedding_features = []
            for pe in camera_embedding:
                camera_embedding_feature = self.camera_encoder(pe)
                camera_embedding_feature = [rearrange(x, '(b f) c h w -> b c f h w', b=bs) for x in camera_embedding_feature]
                camera_embedding_features.append(camera_embedding_feature)
        else:
            bs = camera_embedding.shape[0]
            assert camera_embedding.ndim == 5
            camera_embedding_features = self.camera_encoder(camera_embedding)
            camera_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs)
                                       for x in camera_embedding_features]

        # Duplicate for CFG
        if isinstance(camera_embedding_features[0], list):
            camera_embedding_features_cfg = [[torch.cat([x, x], dim=0) for x in camera_embedding_feature]
                                       for camera_embedding_feature in camera_embedding_features] \
                if do_classifier_free_guidance else camera_embedding_features
        else:
            camera_embedding_features_cfg = [torch.cat([x, x], dim=0) for x in camera_embedding_features] \
                if do_classifier_free_guidance else camera_embedding_features
        
        print("Camera features ready", file=sys.stderr, flush=True)
        # ==================== END CAMERA ENCODING ====================

        # ==================== ENCODE INVERSION CAMERA FEATURES FIRST ====================
        print("Encoding inversion camera embeddings...", file=sys.stderr, flush=True)
        if isinstance(inversion_camera_embedding, list):
            assert all([x.ndim == 5 for x in inversion_camera_embedding])
            inversion_bs = inversion_camera_embedding[0].shape[0]
            inversion_camera_embedding_features = []
            for pe in inversion_camera_embedding:
                inversion_camera_embedding_features = self.camera_encoder(pe)
                inversion_camera_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs) for x in inversion_camera_embedding_features]
                inversion_camera_embedding_features.append(inversion_camera_embedding_features)
        else:
            bs = camera_embedding.shape[0]
            assert camera_embedding.ndim == 5
            camera_embedding_features = self.camera_encoder(camera_embedding)
            inversion_camera_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs)
                                       for x in camera_embedding_features]

        # Duplicate for CFG
        if isinstance(inversion_camera_embedding_features[0], list):
            inversion_camera_embedding_features_cfg = [[torch.cat([x, x], dim=0) for x in inversion_camera_embedding_feature]
                                       for inversion_camera_embedding_feature in inversion_camera_embedding_features] \
                if do_classifier_free_guidance else inversion_camera_embedding_features
        else:
            inversion_camera_embedding_features_cfg = [torch.cat([x, x], dim=0) for x in inversion_camera_embedding_features] \
                if do_classifier_free_guidance else inversion_camera_embedding_features
        
        print("Inversion Camera features ready", file=sys.stderr, flush=True)
        # ==================== END CAMERA ENCODING ====================




        # ==================== PREPARE LATENTS (WITH OPTIONAL INVERSION) ====================
        print(f"Preparing latents (inversion={use_inversion})...", file=sys.stderr, flush=True)

        if use_inversion and input_image_path:
            print(f"Using SOLUTION 1: Frame 0 (Inverted) + Frames 1-{video_length-1} (Random)", file=sys.stderr, flush=True)

            # Extract first frame camera features for inversion
            if isinstance(inversion_camera_embedding_features_cfg[0], list):
                inversion_camera_features_single = [[x[:, :, :1] for x in feature_list] 
                                                   for feature_list in inversion_camera_embedding_features_cfg]
            else:
                inversion_camera_features_single = [x[:, :, :1] for x in inversion_camera_embedding_features_cfg]
            
            # 1. Invert SINGLE frame (frame 0)
            concatenate_threshold = 4
            inverted_frame0 = self.invert_latents_from_image(
                input_image_path,
                5,  # Only 1 frame
                height,
                width,
                inversion_text_embeddings,
                inversion_camera_features_single,  # Only first frame
                device,
                1.5,
                num_inversion_steps
            )
            print(f"Inverted frame 0 shape: {inverted_frame0.shape}", file=sys.stderr, flush=True)
            print(f"Inverted frame 0 stats - min: {inverted_frame0.min().item():.4f}, max: {inverted_frame0.max().item():.4f}, mean: {inverted_frame0.mean().item():.4f}", file=sys.stderr, flush=True)

            # 2. Generate random noise for remaining frames (1 to video_length-1)
            random_frames = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length - 0,  # Frames 1 to end
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                None,
            )
            print(f"Random frames shape (1-{video_length-1}): {random_frames.shape}", file=sys.stderr, flush=True)
            print(f"Random frames stats - min: {random_frames.min().item():.4f}, max: {random_frames.max().item():.4f}, mean: {random_frames.mean().item():.4f}", file=sys.stderr, flush=True)

            # weights_A = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).view(1, 1, 5, 1, 1).to(device)
            # weights_B = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]).view(1, 1, 5, 1, 1).to(device)
            # latents = inverted_frame0 * weights_A + random_frames * weights_B



            # 3. Concatenate: [inverted frame 0] + [random frames 1-4]
            # latents = torch.cat([inverted_frame0, random_frames], dim=2)

            latents = inverted_frame0
            
            print(f"✓ Hybrid latents created: {latents.shape}", file=sys.stderr, flush=True)
            print(f"  - Frame 0: Inverted from input image", file=sys.stderr, flush=True)
            print(f"  - Frames 1-{video_length-1}: Random noise", file=sys.stderr, flush=True)
        else:
            # Use random noise (original behavior)
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
            )
            print(f"Using random noise initialization", file=sys.stderr, flush=True)

        print(f"Final latents shape: {latents.shape}", file=sys.stderr, flush=True)
        print(f"Final latents stats - min: {latents.min().item():.4f}, max: {latents.max().item():.4f}, mean: {latents.mean().item():.4f}", file=sys.stderr, flush=True)
        # ==================== END LATENTS PREPARATION ====================
        
        latents_dtype = latents.dtype
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop (UNCHANGED from original)
        print("Starting denoising loop...", file=sys.stderr, flush=True)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if i <= 15:
                  continue
                
                noise_pred_full = torch.zeros_like(latents).to(latents.device)
                mask_full = torch.zeros_like(latents).to(latents.device)
                noise_preds = []
                for multidiff_step in range(multidiff_total_steps):
                    start_idx = multidiff_step * (single_model_length - multidiff_overlaps)
                    latent_partial = latents[:, :, start_idx: start_idx + single_model_length].contiguous()
                    mask_full[:, :, start_idx: start_idx + single_model_length] += 1

                    if isinstance(camera_embedding, list):
                        camera_embedding_features_input = camera_embedding_features_cfg[multidiff_step]
                    else:
                        camera_embedding_features_input = [x[:, :, start_idx: start_idx + single_model_length]
                                                         for x in camera_embedding_features_cfg]

                    latent_model_input = torch.cat([latent_partial] * 2) if do_classifier_free_guidance else latent_partial
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings,
                                           camera_embedding_features=camera_embedding_features_input).sample.to(dtype=latents_dtype)
                    
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_preds.append(noise_pred)
                    
                for pred_idx, noise_pred in enumerate(noise_preds):
                    start_idx = pred_idx * (single_model_length - multidiff_overlaps)
                    noise_pred_full[:, :, start_idx: start_idx + single_model_length] += noise_pred / mask_full[:, :, start_idx: start_idx + single_model_length]

                latents = self.scheduler.step(noise_pred_full, t, latents, **extra_step_kwargs).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing (UNCHANGED)
        print("Decoding video...", file=sys.stderr, flush=True)
        video = self.decode_latents(latents)

        if output_type == "tensor":
            video = torch.from_numpy(video)

        print("✓ Pipeline complete\n", file=sys.stderr, flush=True)
        
        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)