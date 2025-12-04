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

    # ==================== NULL TEXT INVERSION METHOD ====================
    def null_text_inversion(
        self,
        image_path: str,
        video_length: int,
        height: int,
        width: int,
        prompt: str,
        camera_embedding_features: List[torch.FloatTensor],
        device: str = "cuda",
        guidance_scale: float = 7.5,
        num_inversion_steps: int = 50,
        num_inner_steps: int = 10,
        learning_rate: float = 1e-3,
    ):
        """
        Perform null text inversion: optimize unconditional embeddings during DDIM inversion.
        Returns: (inverted_latents, optimized_uncond_embeddings_list)
        """
        from PIL import Image
        import torchvision.transforms as T
        import sys
        
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"NULL TEXT INVERSION Started", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)
        
        # Load and encode image to latent
        print(f"Loading image: {image_path}", file=sys.stderr, flush=True)
        image = Image.open(image_path).convert("RGB")
        
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        img_tensor = transform(image).unsqueeze(0).to(device=device)
        
        with torch.no_grad():
            latent = self.vae.encode(img_tensor).latent_dist.sample()
            latent = latent * 0.18215
            latent = latent.detach()  # Ensure no gradients
        
        # Expand to video (repeat same frame)
        latent_5d = latent.unsqueeze(2).repeat(1, 1, video_length, 1, 1)
        print(f"Encoded latent shape: {latent_5d.shape}", file=sys.stderr, flush=True)
        
        # Get conditional (prompt) embeddings - these stay fixed
        with torch.no_grad():
            text_embeddings_cond = self._encode_prompt(
                [prompt], device, 1, False, None
            )
            text_embeddings_cond = text_embeddings_cond.detach()
        print(f"Conditional embedding shape: {text_embeddings_cond.shape}", file=sys.stderr, flush=True)
        
        # Initialize unconditional embeddings - these will be optimized
        with torch.no_grad():
            uncond_embeddings = self._encode_prompt(
                [""], device, 1, False, None
            )
            uncond_embeddings = uncond_embeddings.detach()
        print(f"Initial unconditional embedding shape: {uncond_embeddings.shape}", file=sys.stderr, flush=True)
        
        # Setup for inversion
        self.scheduler.set_timesteps(num_inversion_steps, device=device)
        timesteps = list(reversed(self.scheduler.timesteps))
        
        x = latent_5d.clone()
        optimized_uncond_list = []  # Store optimized uncond embeddings per timestep
        
        print(f"Starting inversion with optimization...", file=sys.stderr, flush=True)
        
        for i, t in enumerate(timesteps):
            print(f"\n--- Timestep {i}/{len(timesteps)}, t={t.item()} ---", file=sys.stderr, flush=True)
            
            # ============ INNER OPTIMIZATION LOOP ============
            # Optimize unconditional embedding to minimize reconstruction error
            uncond_embeddings_opt = uncond_embeddings.clone().detach()
            uncond_embeddings_opt.requires_grad = True
            
            optimizer = torch.optim.Adam([uncond_embeddings_opt], lr=learning_rate)
            
            x_prev = x.detach().clone()
            
            for inner_step in range(num_inner_steps):
                optimizer.zero_grad()
                
                # Prepare CFG input
                latent_model_input = torch.cat([x_prev] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Combined embeddings for CFG
                text_embeddings_cfg = torch.cat([uncond_embeddings_opt, text_embeddings_cond])
                
                # Predict noise with current unconditional embedding - ENABLE GRADIENTS
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings_cfg,
                    camera_embedding_features=camera_embedding_features
                ).sample
                
                # Apply CFG
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute predicted x0
                current_t_val = t.item()
                alpha_t = self.scheduler.alphas_cumprod[current_t_val]
                predicted_x0 = (x_prev - (1 - alpha_t).sqrt() * noise_pred_cfg) / alpha_t.sqrt()
                
                # Loss: reconstruction error between predicted x0 and actual latent
                loss = torch.nn.functional.mse_loss(predicted_x0, latent_5d)
                
                loss.backward()
                optimizer.step()
                
                if inner_step % 5 == 0 or inner_step == num_inner_steps - 1:
                    print(f"  Inner step {inner_step}/{num_inner_steps}, Loss: {loss.item():.6f}", 
                          file=sys.stderr, flush=True)
            
            # Store optimized unconditional embedding for this timestep
            optimized_uncond_list.append(uncond_embeddings_opt.detach().clone())
            print(f"✓ Optimized uncond embedding for timestep {i}", file=sys.stderr, flush=True)
            
            # ============ PERFORM INVERSION STEP WITH OPTIMIZED EMBEDDING ============
            with torch.no_grad():
                latent_model_input = torch.cat([x] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                text_embeddings_cfg = torch.cat([uncond_embeddings_opt, text_embeddings_cond])
                
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings_cfg,
                    camera_embedding_features=camera_embedding_features
                ).sample
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # DDIM inversion step
                alpha_t = self.scheduler.alphas_cumprod[current_t_val]
                
                if i + 1 < len(timesteps):
                    next_t_val = timesteps[i + 1].item()
                else:
                    next_t_val = self.scheduler.timesteps[0].item()
                
                alpha_t_next = self.scheduler.alphas_cumprod[next_t_val]
                
                predicted_x0 = (x - (1 - alpha_t).sqrt() * noise_pred_cfg) / alpha_t.sqrt()
                x = alpha_t_next.sqrt() * predicted_x0 + (1 - alpha_t_next).sqrt() * noise_pred_cfg
            
            print(f"Latent stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}", 
                  file=sys.stderr, flush=True)
        
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"✓ Null text inversion complete", file=sys.stderr, flush=True)
        print(f"Inverted latent shape: {x.shape}", file=sys.stderr, flush=True)
        print(f"Optimized {len(optimized_uncond_list)} unconditional embeddings", file=sys.stderr, flush=True)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)
        
        return x, optimized_uncond_list
    # ==================== END NULL TEXT INVERSION ====================

    # Keep original DDIM inversion as fallback
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
        Perform standard DDIM inversion from an input image.
        Returns inverted latents at maximum noise level.
        """
        from PIL import Image
        import torchvision.transforms as T
        import sys
        
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"DDIM Inversion Started", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)
        
        # Load and preprocess image
        print(f"Loading image: {image_path}", file=sys.stderr, flush=True)
        image = Image.open(image_path).convert("RGB")
        
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        img_tensor = transform(image).unsqueeze(0).to(device=device, dtype=text_embeddings.dtype)
        print(f"Image tensor shape: {img_tensor.shape}", file=sys.stderr, flush=True)
        
        # Encode to latent space
        with torch.no_grad():
            latent = self.vae.encode(img_tensor).latent_dist.sample()
            latent = latent * 0.18215
        
        print(f"Encoded latent shape: {latent.shape}", file=sys.stderr, flush=True)
        print(f"Latent stats - min: {latent.min().item():.4f}, max: {latent.max().item():.4f}, mean: {latent.mean().item():.4f}", file=sys.stderr, flush=True)
        
        # Expand to video (repeat same frame)
        latent_5d = latent.unsqueeze(2).repeat(1, 1, video_length, 1, 1)
        print(f"Expanded to 5D: {latent_5d.shape}", file=sys.stderr, flush=True)
        
        # DDIM inversion - go from x_0 to x_T
        self.scheduler.set_timesteps(num_inversion_steps, device=device)
        timesteps = list(reversed(self.scheduler.timesteps))
        print(f"Inversion timesteps: {len(timesteps)}", file=sys.stderr, flush=True)
        print(f"First timestep: {timesteps[0]}, Last timestep: {timesteps[-1]}", file=sys.stderr, flush=True)
        
        x = latent_5d.clone()
        do_cfg = guidance_scale > 1.0
        
        for i, t in enumerate(timesteps):
            if i % 10 == 0:
                print(f"  Inversion step {i}/{len(timesteps)}, t={t.item()}", file=sys.stderr, flush=True)
                print(f"    Latent stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}", file=sys.stderr, flush=True)
            
            if do_cfg:
                latent_model_input = torch.cat([x] * 2)
            else:
                latent_model_input = x
            
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    camera_embedding_features=camera_embedding_features
                ).sample
            
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            current_t_val = t.item()
            alpha_t = self.scheduler.alphas_cumprod[current_t_val]
            
            if i + 1 < len(timesteps):
                next_t_val = timesteps[i + 1].item()
            else:
                next_t_val = self.scheduler.timesteps[0].item()
            
            alpha_t_next = self.scheduler.alphas_cumprod[next_t_val]
            
            predicted_x0 = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            predicted_x0 = predicted_x0.detach()
            
            x = alpha_t_next.sqrt() * predicted_x0 + (1 - alpha_t_next).sqrt() * noise_pred
            del predicted_x0, noise_pred
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        print(f"✓ Inversion complete: {x.shape}", file=sys.stderr, flush=True)
        print(f"Final inverted latent stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}", file=sys.stderr, flush=True)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)
        return x

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
        input_image_path: Optional[str] = None,
        use_inversion: bool = False,
        use_null_text_inversion: bool = False,
        num_inversion_steps: int = 100,
        null_text_inner_steps: int = 10,
        null_text_learning_rate: float = 1e-3,
        **kwargs,
    ):
        import sys
        
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"GenPhotoPipeline __call__", file=sys.stderr, flush=True)
        print(f"use_inversion: {use_inversion}", file=sys.stderr, flush=True)
        print(f"use_null_text_inversion: {use_null_text_inversion}", file=sys.stderr, flush=True)
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

        # Encode camera embeddings
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

        if isinstance(camera_embedding_features[0], list):
            camera_embedding_features_cfg = [[torch.cat([x, x], dim=0) for x in camera_embedding_feature]
                                       for camera_embedding_feature in camera_embedding_features] \
                if do_classifier_free_guidance else camera_embedding_features
        else:
            camera_embedding_features_cfg = [torch.cat([x, x], dim=0) for x in camera_embedding_features] \
                if do_classifier_free_guidance else camera_embedding_features
        
        print("Camera features ready", file=sys.stderr, flush=True)

        # Encode inversion camera embeddings
        print("Encoding inversion camera embeddings...", file=sys.stderr, flush=True)
        if isinstance(inversion_camera_embedding, list):
            assert all([x.ndim == 5 for x in inversion_camera_embedding])
            inversion_bs = inversion_camera_embedding[0].shape[0]
            inversion_camera_embedding_features = []
            for pe in inversion_camera_embedding:
                inversion_camera_embedding_feature = self.camera_encoder(pe)
                inversion_camera_embedding_feature = [rearrange(x, '(b f) c h w -> b c f h w', b=inversion_bs) for x in inversion_camera_embedding_feature]
                inversion_camera_embedding_features.append(inversion_camera_embedding_feature)
        else:
            inversion_bs = inversion_camera_embedding.shape[0]
            assert inversion_camera_embedding.ndim == 5
            inversion_camera_embedding_features = self.camera_encoder(inversion_camera_embedding)
            inversion_camera_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=inversion_bs)
                                       for x in inversion_camera_embedding_features]

        if isinstance(inversion_camera_embedding_features[0], list):
            inversion_camera_embedding_features_cfg = [[torch.cat([x, x], dim=0) for x in inversion_camera_embedding_feature]
                                       for inversion_camera_embedding_feature in inversion_camera_embedding_features] \
                if do_classifier_free_guidance else inversion_camera_embedding_features
        else:
            inversion_camera_embedding_features_cfg = [torch.cat([x, x], dim=0) for x in inversion_camera_embedding_features] \
                if do_classifier_free_guidance else inversion_camera_embedding_features
        
        print("Inversion Camera features ready", file=sys.stderr, flush=True)

        # Store optimized uncond embeddings for generation
        optimized_uncond_embeddings_list = None

        # Prepare latents with optional null text inversion
        print(f"Preparing latents (null_text={use_null_text_inversion}, inversion={use_inversion})...", file=sys.stderr, flush=True)

        if use_null_text_inversion and input_image_path:
            print(f"Using NULL TEXT INVERSION: Optimizing unconditional embeddings", file=sys.stderr, flush=True)

            # Extract first frame camera features for inversion
            if isinstance(inversion_camera_embedding_features_cfg[0], list):
                inversion_camera_features_single = [[x[:, :, :1] for x in feature_list] 
                                                   for feature_list in inversion_camera_embedding_features_cfg]
            else:
                inversion_camera_features_single = [x[:, :, :1] for x in inversion_camera_embedding_features_cfg]
            
            # Perform null text inversion
            inverted_frame0, optimized_uncond_embeddings_list = self.null_text_inversion(
                input_image_path,
                1,  # Single frame
                height,
                width,
                prompt[0] if isinstance(prompt, list) else prompt,
                inversion_camera_features_single,
                device,
                guidance_scale,
                num_inversion_steps,
                null_text_inner_steps,
                null_text_learning_rate
            )
            
            print(f"Inverted frame 0 shape: {inverted_frame0.shape}", file=sys.stderr, flush=True)

            # Generate random noise for remaining frames
            random_frames = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length - 1,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                None,
            )
            
            latents = torch.cat([inverted_frame0, random_frames], dim=2)
            print(f"✓ Hybrid latents with null text inversion: {latents.shape}", file=sys.stderr, flush=True)

        elif use_inversion and input_image_path:
            print(f"Using standard DDIM inversion", file=sys.stderr, flush=True)

            if isinstance(inversion_camera_embedding_features_cfg[0], list):
                inversion_camera_features_single = [[x[:, :, :1] for x in feature_list] 
                                                   for feature_list in inversion_camera_embedding_features_cfg]
            else:
                inversion_camera_features_single = [x[:, :, :1] for x in inversion_camera_embedding_features_cfg]
            
            inverted_frame0 = self.invert_latents_from_image(
                input_image_path,
                5,
                height,
                width,
                inversion_text_embeddings,
                inversion_camera_features_single,
                device,
                guidance_scale,
                num_inversion_steps
            )
            
            random_frames = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length - 5,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                None,
            )
            
            latents = torch.cat([inverted_frame0, random_frames], dim=2)
            print(f"✓ Hybrid latents with DDIM inversion: {latents.shape}", file=sys.stderr, flush=True)
        else:
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

        print(f"Final latents ready: {latents.shape}", file=sys.stderr, flush=True)
        print(f"Latents stats - min: {latents.min().item():.4f}, max: {latents.max().item():.4f}, mean: {latents.mean().item():.4f}", file=sys.stderr, flush=True)
        # ==================== END LATENTS PREPARATION ====================
        
        latents_dtype = latents.dtype
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop (UNCHANGED from original)
        print("Starting denoising loop...", file=sys.stderr, flush=True)
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