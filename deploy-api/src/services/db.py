from src.core.config import get_settings

def _get_repo():
    if get_settings().effective_use_memory_repo:
        import src.services.memory_repo as r
    else:
        import src.services.firestore_repo as r
    return r

def get_firestore_client(*args, **kwargs):
    return _get_repo().get_firestore_client(*args, **kwargs)

def append_log(*args, **kwargs):
    return _get_repo().append_log(*args, **kwargs)

def get_deployment(*args, **kwargs):
    return _get_repo().get_deployment(*args, **kwargs)

def update_deployment(*args, **kwargs):
    return _get_repo().update_deployment(*args, **kwargs)
