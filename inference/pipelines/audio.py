"""Text-to-Audio pipeline."""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

import torch

from pipelines.base import BasePipeline


class AudioPipeline(BasePipeline):
    """Generic Diffusers TextToAudio pipeline."""

    def load(self) -> None:
        from diffusers import AutoPipelineForText2Audio

        self._pipeline = AutoPipelineForText2Audio.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            token=self.token,
        )
        self._pipeline.to(self.device)

    def run(
        self,
        prompt: str,
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        audio_length_in_s: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded")
            
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        kwargs_dict = {}
        if audio_length_in_s is not None:
            kwargs_dict["audio_length_in_s"] = audio_length_in_s
            
        out = self._pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs_dict
        )
        
        audio = out.audios[0]
        import soundfile as sf
        
        tmp_path = f"/tmp/{uuid.uuid4().hex}.wav"
        # Often the pipeline has self._pipeline.unet.config.sample_rate or pipeline.vocoder.config.sampling_rate
        # For AudioLDM2 it's 16000
        sample_rate = 16000
        if hasattr(self._pipeline, "vocoder") and hasattr(self._pipeline.vocoder, "config"):
            sample_rate = getattr(self._pipeline.vocoder.config, "sampling_rate", 16000)
            
        sf.write(tmp_path, audio.T, sample_rate)

        return {
            "file_path": tmp_path,
            "file_extension": ".wav",
            "content_type": "audio/wav",
            "model_id": self.model_id,
            "seed": seed,
            "duration": audio_length_in_s
        }
