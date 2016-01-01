import os

_sound_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds')

def sound_path(file_name):
    """locates a demo sound file"""
    return os.path.join(_sound_dir, file_name)
