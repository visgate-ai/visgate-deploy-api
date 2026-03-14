"""Text-to-Video pipeline."""

import os
import uuid
from typing import Any, Optional

import torch

from pipelines.base import BasePipeline


class VideoPipeline(BasePipeline):
    """Generic Diffusers TextToVideo pipeline."""

    def load(self) -> None:
        from diffusers import AutoPipelineForText2Video

        # Memory efficient loading
        self._pipeline = AutoPipelineForText2Video.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            token=self.token,
        )
        self._pipeline.to(self.device)
        try:
            self._pipeline.enable_model_cpu_offload()
        except AttributeError:
            try:
                self._pipeline.enable_vae_slicing()
            except AttributeError:
                pass

    def run(
        self,
        prompt: str,
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
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
            
        # Call diffusers
        out = self._pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            output_type="np", # Get numpy array to export to video properly
            **kwargs,
        )
        
        frames = out.frames if hasattr(out, "frames") else out[0]
        if not len(frames):
            return {"error": "No frames generated", "model_id": self.model_id}
            
        from diffusers.utils import export_to_video
        
        import tempfile
        # Use tempfile for thread safety
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
        export_to_video(frames, tmp_path, fps=8)

        return {
            "file_path": tmp_path,
            "file_extension": ".mp4",
            "content_type": "video/mp4",
            "model_id": self.model_id,
            "seed": seed,
            "num_frames": len(frames)
        }
