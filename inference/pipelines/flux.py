"""Flux (black-forest-labs) diffusion pipeline."""

import base64
import io
from typing import Any, Optional

import torch

from pipelines.base import BasePipeline


class FluxPipeline(BasePipeline):
    """Flux 1.0 Schnell / Dev via diffusers."""

    def load(self) -> None:
        try:
            from diffusers import FluxPipeline as DiffusersFluxPipeline
        except ImportError:
            from diffusers import AutoPipelineForText2Image as DiffusersFluxPipeline
        self._pipeline = DiffusersFluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            token=self.token,
        )
        self._pipeline.to(self.device)
        # Optional: enable memory optimizations
        try:
            self._pipeline.enable_attention_slicing()
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
        # Flux schnell typically 512x512 or 1024x1024
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
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {
            "image_base64": b64,
            "model_id": self.model_id,
            "seed": seed,
            "height": height,
            "width": width,
        }
