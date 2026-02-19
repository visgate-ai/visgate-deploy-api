"""SDXL and Stable Diffusion XL pipeline via diffusers AutoPipeline."""

import base64
import io
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
