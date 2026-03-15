"""Flux (black-forest-labs) diffusion pipeline."""

import base64
import io
import os
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

        use_local_files = os.path.isdir(self.model_id)
        has_model_index = os.path.exists(os.path.join(self.model_id, "model_index.json")) if use_local_files else False
        has_config = os.path.exists(os.path.join(self.model_id, "config.json")) if use_local_files else False
        if self.device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif self.device.startswith("cuda"):
            dtype = torch.float16
        else:
            dtype = torch.float32

        self._log(
            "INFO",
            (
                f"load start source={self.model_id} source_kind={self._describe_source()} "
                f"device={self.device} dtype={dtype} local_files_only={use_local_files} "
                f"model_index={has_model_index} config={has_config}"
            ),
        )
        self._log("INFO", "Calling Flux from_pretrained")

        self._pipeline = DiffusersFluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            token=self.token,
            local_files_only=use_local_files,
        )
        self._log("INFO", "from_pretrained completed")
        self._log("INFO", f"Moving pipeline to device {self.device}")
        self._pipeline.to(self.device)
        self._log("INFO", "Pipeline moved to target device")
        # Memory optimizations (best-effort; ignore if unsupported)
        enabled_optimizations: list[str] = []
        for method in (
            "enable_vae_slicing",
            "enable_vae_tiling",
            "enable_attention_slicing",
        ):
            try:
                getattr(self._pipeline, method)()
                enabled_optimizations.append(method)
            except Exception:
                pass
        # xformers memory-efficient attention (faster if installed)
        xformers_enabled = False
        try:
            self._pipeline.enable_xformers_memory_efficient_attention()
            xformers_enabled = True
        except Exception:
            pass
        self._log(
            "INFO",
            (
                f"Load complete optimizations={enabled_optimizations or ['none']} "
                f"xformers={xformers_enabled}"
            ),
        )

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
