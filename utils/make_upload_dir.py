from os import path, makedirs
from shutil import rmtree


def make_upload_dir(path_: str):
    if path.exists(path_):
        rmtree(path=path_, ignore_errors=True)
    makedirs(path_)
