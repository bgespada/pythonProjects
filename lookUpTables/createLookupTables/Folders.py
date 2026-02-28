import os
from pathlib import Path


def appFolder():
    return Path(__file__).parent.absolute()

def folder(path):
    f = f"{path}"
    if not os.path.exists(f):
        os.makedirs(f)
    return f
