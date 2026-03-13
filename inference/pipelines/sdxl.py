"""SDXL and Stable Diffusion XL pipeline via diffusers AutoPipeline."""

from __future__ import annotations

import base64
import io
import os
import uuid
from typing import Any, Optional

import torch

from pipelines.base import BasePipeline


class SDXLPipeline(BasePipeline):
    """SDXL / SD Turbo via diffusers AutoPipelineForText2Image."""

    def load(self) -> None:
        from diffusers import AutoPipelineForText2Image

        self._pipeline = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            token=self.token,
        )
        self._pipeline.to(self.device)
        # Memory optimizations (best-effort)
        for method in (
            "enable_vae_slicing",
            "enable_vae_tiling",
            "enable_attention_slicing",
        ):
            try:
                getattr(self._pipeline, method)()
            except Exception:
                pass
        # xformers memory-efficient attention (faster if installed)
        try:
            self._pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    def run(
        self,
        prompt: str,
        *,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        height = height or 1024
        width = width or 1024
        out = self._pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            **kwargs,
        )
        images = out.images
        if not images:
            return {"error": "No image generated", "model_id": self.model_id}
        img = images[0]
        
        tmp_path = f"/tmp/{uuid.uuid4().hex}.png"
        img.save(tmp_path, format="PNG")
        
        return {
            "file_path": tmp_path,
            "file_extension": ".png",
            "content_type": "image/png",
            "model_id": self.model_id,
            "seed": seed,
            "height": height,
            "width": width,
        }
