from os import path
from shutil import rmtree


def delete_file(path_: str):
    if path.exists(path_):
        rmtree(path=path_, ignore_errors=True)
