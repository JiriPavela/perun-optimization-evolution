""" Module containing load and save methods for various formats, such as JSON, pickle, bz, etc.
"""
import os
import sys
import json
import zlib
import pickle
import bz2
from typing import Dict, List, Any, Optional


# JSON type alias
JsonType = Dict[str, Any]


# Constants
# Directory containing saved experiments
EXPERIMENTS_LOC = 'storage'
# BZ2 suffix string
BZ2_SUFFIX = '.bz2'
# Maximum increments of pickle recursion limit
MAX_LIMIT_INC = 5000


def load_and_deflate(file_name: str, dir_path: str) -> JsonType:
    """ Load and decompress JSON file compressed with bz2.

    :param file_name: file name (with suffix) of the compressed file
    :param dir_path: path to the directory containing the requested file

    :return: file contents in the JSON format
    """
    # Open -> read -> decompress -> JSON load
    with open(os.path.join(dir_path, file_name), 'rb') as compressed:
        return json.loads(zlib.decompressobj().decompress(compressed.read()).decode())


def find_project_stats(project_name: str, dir_path: str, prefix: str) -> List[str]:
    """ Lookup Dynamic stats files based on the project name, directory and prefix.

    :param project_name: name of the project that the Dynamic stats are connected to
    :param dir_path: path to the directory containing the Dynamic stats
    :param prefix: prefix of the Dynamic stats files

    :return: list of the file paths
    """
    return [f for f in os.listdir(dir_path) if f.startswith(f'{prefix}_{project_name}')]


def save_object(obj: Any, name: Optional[str], directory: str) -> None:
    """ Serialize, compress and save an object.

    :param obj: the object to serialize. We do not impose any type restrictions on the saved object
    :param name: name of the resulting file (excluding the compression suffix)
    :param directory: the target directory
    """
    # Handle edge case
    if name is None:
        return
    # Make sure that the directory exists before trying to save a file there
    if not os.path.exists(directory):
        os.mkdir(directory)
    # Create the file
    with open(os.path.join(directory, name + BZ2_SUFFIX), 'wb') as pk:
        # There may be a serialization issue due to the python recursion limit - depending on the
        # complexity of the nested structures within the object = retry as many times as possible
        while True:
            try:
                # Attempt to save the object
                pk.write(bz2.compress(pickle.dumps(obj, -1)))
                # Success 
                break
            except RecursionError:
                # Increase the recursion limit 
                limit = sys.getrecursionlimit()
                sys.setrecursionlimit(limit + min(limit, MAX_LIMIT_INC))
                print(
                    f"Recursion limit reached: increasing recursion limit from {limit}'\
                    ' to {sys.getrecursionlimit()}"
                )


def load_object(path: str) -> Any:
    """ Open, decompress and deserialize an object. No type restrictions are assumed.

    :param path: path to the file to load

    :return: deserialized object
    """
    with open(path, 'rb') as pk:
        return pickle.loads(bz2.decompress(pk.read()))


def list_saves(directory: str) -> List[str]:
    """ List files containing saved object (presumably using the 'save_object' function) in the 
    specified directory.

    :param directory: directory to search in

    :return: list of files presumably containing saved objects
    """
    if not os.path.exists(directory):
        os.mkdir(directory)
    return [f for f in os.listdir(directory) if f.endswith(BZ2_SUFFIX)]
