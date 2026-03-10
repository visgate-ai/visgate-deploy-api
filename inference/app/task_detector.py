def detect_task(model_id: str, environ_task: str = "") -> str:
    environ_task = environ_task.strip().lower()
    if environ_task in ["text2img", "image2img", "text2video", "video2video", "speech_to_text", "text_to_speech"]:
        return environ_task

    model_id_lower = model_id.lower()
    
    # Video Models
    if any(token in model_id_lower for token in ["wan2", "cogvideo", "svd", "animatediff", "t2v"]):
        return "text2video"
        
    # Audio Models
    if any(token in model_id_lower for token in ["whisper", "wav2vec", "asr"]):
        return "speech_to_text"
    if any(token in model_id_lower for token in ["tts", "bark", "speecht5", "xtts", "audioldm"]):
        return "text_to_speech"
        
    # Default to Image (largest category)
    return "text2img"
