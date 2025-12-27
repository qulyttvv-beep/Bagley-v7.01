"""
🎨 BagleyImagePipeline - Complete Image Generation Pipeline
Integrates DiT, VAE, and text encoders for end-to-end generation
"""

import math
from typing import Optional, List, Union, Callable, Any
from dataclasses import dataclass
import logging

import torch
import torch.nn.functional as F
from torch import Tensor

from bagley.models.image.model import BagleyDiT, BagleyDiTConfig
from bagley.models.image.vae import BagleyVAE

logger = logging.getLogger(__name__)


@dataclass
class ImageGenerationConfig:
    """Configuration for image generation"""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 28
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    num_images: int = 1


class RectifiedFlowScheduler:
    """
    Rectified Flow Scheduler for sampling
    
    Implements the rectified flow ODE for faster, higher quality sampling.
    Based on FLUX.1's approach.
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 28,
        shift: float = 3.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.shift = shift
        
        self.timesteps = None
        self.sigmas = None
    
    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """Prepare timesteps for inference"""
        self.num_inference_steps = num_inference_steps
        
        # Shifted timesteps for rectified flow
        timesteps = torch.linspace(1, 0, num_inference_steps + 1, device=device)
        
        # Apply shift (increases SNR at high noise levels)
        timesteps = timesteps / (timesteps + self.shift * (1 - timesteps))
        
        self.timesteps = timesteps[:-1]  # Exclude final 0
        self.sigmas = timesteps
    
    def step(
        self,
        model_output: Tensor,
        timestep_idx: int,
        sample: Tensor,
    ) -> Tensor:
        """Single denoising step using Euler method"""
        sigma = self.sigmas[timestep_idx]
        sigma_next = self.sigmas[timestep_idx + 1] if timestep_idx + 1 < len(self.sigmas) else 0.0
        
        dt = sigma_next - sigma
        
        # Euler step: x_t+1 = x_t + dt * v_t
        prev_sample = sample + dt * model_output
        
        return prev_sample
    
    def add_noise(
        self,
        original_samples: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """Add noise according to rectified flow schedule"""
        sigmas = timesteps.view(-1, 1, 1, 1)
        noisy_samples = (1 - sigmas) * original_samples + sigmas * noise
        return noisy_samples


class BagleyImagePipeline:
    """
    🎨 Complete Image Generation Pipeline
    
    Orchestrates:
    - Text encoding (T5-XXL + CLIP)
    - DiT denoising
    - VAE decoding
    - Classifier-free guidance
    """
    
    def __init__(
        self,
        dit: Optional[BagleyDiT] = None,
        vae: Optional[BagleyVAE] = None,
        text_encoder: Optional[Any] = None,  # T5-XXL
        clip_encoder: Optional[Any] = None,  # CLIP
        tokenizer: Optional[Any] = None,
        scheduler: Optional[RectifiedFlowScheduler] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self.clip_encoder = clip_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler or RectifiedFlowScheduler()
        self.device = device
        self.dtype = dtype
        
        logger.info("Initialized BagleyImagePipeline")
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> "BagleyImagePipeline":
        """Load complete pipeline from pretrained"""
        import os
        
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        
        # Load DiT
        dit = BagleyDiT.from_pretrained(os.path.join(path, "dit"), device, dtype)
        
        # Load VAE
        vae = BagleyVAE.from_pretrained(os.path.join(path, "vae"), device)
        
        # Text encoders would be loaded here
        # For now, they're placeholders
        
        return cls(
            dit=dit,
            vae=vae,
            device=device,
            dtype=torch_dtype,
        )
    
    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        do_classifier_free_guidance: bool = True,
    ) -> tuple:
        """Encode text prompts"""
        # Placeholder - would use T5-XXL and CLIP
        # Returns encoder_hidden_states and pooled_embeds
        
        batch_size = 2 if do_classifier_free_guidance else 1
        max_len = 512
        hidden_size = 4096
        
        # Placeholder tensors
        encoder_hidden_states = torch.randn(
            batch_size, max_len, hidden_size,
            device=self.device, dtype=self.dtype
        )
        pooled_embeds = torch.randn(
            batch_size, hidden_size,
            device=self.device, dtype=self.dtype
        )
        
        return encoder_hidden_states, pooled_embeds
    
    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Prepare initial noise latents"""
        latent_channels = self.dit.config.in_channels if self.dit else 16
        vae_scale = 8  # VAE downsampling factor
        
        shape = (
            batch_size,
            latent_channels,
            height // vae_scale,
            width // vae_scale,
        )
        
        latents = torch.randn(
            shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        
        return latents
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = "",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, int, Tensor], None]] = None,
        callback_steps: int = 1,
    ) -> List[Any]:
        """
        Generate images from text prompts.
        
        Args:
            prompt: Text prompt(s) for generation
            negative_prompt: Negative prompt(s)
            height: Output image height
            width: Output image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            num_images_per_prompt: Images per prompt
            seed: Random seed for reproducibility
            callback: Progress callback function
            callback_steps: Callback frequency
            
        Returns:
            List of generated images (PIL or tensor)
        """
        # Handle prompt list
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        
        # Set up generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Encode prompts
        do_cfg = guidance_scale > 1.0
        encoder_hidden_states, pooled_embeds = self.encode_prompt(
            prompt[0],
            negative_prompt if isinstance(negative_prompt, str) else negative_prompt[0],
            do_classifier_free_guidance=do_cfg,
        )
        
        # Prepare latents
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            height,
            width,
            generator,
        )
        
        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        
        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            
            # Get timestep embedding
            timestep = torch.full(
                (latent_model_input.shape[0],),
                t.item(),
                device=self.device,
                dtype=self.dtype,
            )
            
            # Predict noise/velocity
            if self.dit is not None:
                output = self.dit(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states,
                    pooled_embeds,
                )
                model_output = output.sample
            else:
                # Placeholder for when model isn't loaded
                model_output = torch.randn_like(latent_model_input)
            
            # Apply CFG
            if do_cfg:
                noise_pred_uncond, noise_pred_cond = model_output.chunk(2)
                model_output = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Scheduler step
            latents = self.scheduler.step(model_output, i, latents)
            
            # Callback
            if callback is not None and i % callback_steps == 0:
                callback(i, num_inference_steps, latents)
        
        # Decode latents
        if self.vae is not None:
            images = self.vae.decode(latents)
            # Convert to PIL images
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            
            try:
                from PIL import Image
                import numpy as np
                pil_images = []
                for image in images:
                    image = (image * 255).round().astype(np.uint8)
                    pil_images.append(Image.fromarray(image))
                return pil_images
            except ImportError:
                return images.tolist()
        else:
            # Return raw latents if VAE not loaded
            return [latents]
    
    async def generate_async(
        self,
        prompt: str,
        **kwargs,
    ) -> List[Any]:
        """Async wrapper for generation"""
        return self(prompt, **kwargs)
    
    def to(self, device: str) -> "BagleyImagePipeline":
        """Move pipeline to device"""
        self.device = device
        if self.dit:
            self.dit = self.dit.to(device)
        if self.vae:
            self.vae = self.vae.to(device)
        if self.text_encoder:
            self.text_encoder = self.text_encoder.to(device)
        return self
