import os


def strip_file(filePath):
    """Extracts file name without directories and extension."""
    return os.path.splitext(os.path.basename(filePath))[0]


def ensure_directory(dir):
    if len(dir) > 0 and not os.path.isdir(dir) and not os.path.isfile(dir):
        os.makedirs(dir)
