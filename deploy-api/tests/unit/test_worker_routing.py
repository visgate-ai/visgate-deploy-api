from src.core.config import Settings
from src.services.worker_routing import AUDIO_PROFILE, IMAGE_PROFILE, VIDEO_PROFILE, infer_worker_profile, resolve_worker_target


def test_infer_worker_profile_prefers_task() -> None:
    assert infer_worker_profile("black-forest-labs/FLUX.1-schnell", "text_to_image") == IMAGE_PROFILE
    assert infer_worker_profile("openai/whisper-large-v3", "speech_to_text") == AUDIO_PROFILE
    assert infer_worker_profile("Wan-AI/Wan2.1-T2V-1.3B", "text_to_video") == VIDEO_PROFILE


def test_infer_worker_profile_uses_registry_for_exact_audio_and_video_models() -> None:
    assert infer_worker_profile("openai/whisper-large-v3") == AUDIO_PROFILE
    assert infer_worker_profile("Wan-AI/Wan2.1-T2V-1.3B") == VIDEO_PROFILE


def test_infer_worker_profile_falls_back_to_model_heuristics() -> None:
    assert infer_worker_profile("THUDM/CogVideoX-2b") == VIDEO_PROFILE
    assert infer_worker_profile("stabilityai/sdxl-turbo") == IMAGE_PROFILE


def test_resolve_worker_target_uses_profile_specific_template_and_image() -> None:
    settings = Settings(
        runpod_template_id="tpl-default",
        runpod_template_id_image="tpl-image",
        runpod_template_id_audio="tpl-audio",
        runpod_template_id_video="tpl-video",
        docker_image="visgateai/inference:latest",
        docker_image_image="visgateai/inference-image:latest",
        docker_image_audio="visgateai/inference-audio:latest",
        docker_image_video="visgateai/inference-video:latest",
    )
    audio_target = resolve_worker_target(settings, "openai/whisper-large-v3", None)
    video_target = resolve_worker_target(settings, "Wan-AI/Wan2.1-T2V-1.3B", None)
    image_target = resolve_worker_target(settings, "stabilityai/sdxl-turbo", None)

    assert audio_target == {
        "profile": AUDIO_PROFILE,
        "template_id": "tpl-audio",
        "image": "visgateai/inference-audio:latest",
    }
    assert video_target == {
        "profile": VIDEO_PROFILE,
        "template_id": "tpl-video",
        "image": "visgateai/inference-video:latest",
    }
    assert image_target == {
        "profile": IMAGE_PROFILE,
        "template_id": "tpl-image",
        "image": "visgateai/inference-image:latest",
    }


def test_resolve_worker_target_falls_back_to_default_template() -> None:
    settings = Settings(
        runpod_template_id="tpl-default",
        docker_image="visgateai/inference:latest",
        docker_image_audio="visgateai/inference-audio:latest",
    )
    target = resolve_worker_target(settings, "openai/whisper-large-v3", None)
    assert target == {
        "profile": AUDIO_PROFILE,
        "template_id": "tpl-default",
        "image": "visgateai/inference-audio:latest",
    }