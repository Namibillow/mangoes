# -*- coding: utf-8 -*-
"""Utility functions to save mangoes objects"""
import os
import tempfile
import zipfile
import json


def save(obj, attributes_to_persist, path, metadata=None):
    """Save the object at the given path

    Parameters
    ----------
    obj
        object to persists
    attributes_to_persist: list
        list o the names of the attributes of the object to persist
    path: str
        path to a folder or an archive where the object will be stored
    metadata: dict
        metadata to store in a .metadata file with the objects

    Returns
    -------
    str
        path to a folder or an archive
    """
    attributes_names = [name for (name, _) in attributes_to_persist]
    _, file_extension = os.path.splitext(path)

    if file_extension == ".zip":
        save_func = _save_as_archive
    else:
        save_func = _save_in_folder
    save_func(obj, path, attributes_names, metadata)


def load(cls, attributes_to_persist, path):
    """Load an object from the given path

    Parameters
    ----------
    path: str
        path to a folder or an archive

    Returns
    -------
    object of type `cls`
    """
    _, file_extension = os.path.splitext(path)

    if file_extension == ".zip":
        load_func = _load_from_archive
    else:
        load_func = _load_from_folder
    return load_func(cls, path, attributes_to_persist)


def _save_in_folder(obj, path, attributes, metadata=None):
    """Save the object's attributes as files in the local folder corresponding to the path.

    :param obj: object to store
    :param path: string, the path the folder where the embedding files will be stored.
    :param attributes: list of attributes to save.
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)

    if metadata:
        with open(os.path.join(path, ".metadata"), mode="w") as metadata_file:
            metadata_file.write(json.dumps(metadata, indent=4))

    for attribute in attributes:
        obj.__getattribute__(attribute).save(path, name=attribute)


def _load_from_folder(cls, path, attributes):
    """Load an object instance from the files in the specified folder.

    :param cls: class of the object to load
    :param path: string, path to the folder where the object instance files are located
    :param attributes: list of tuples defining attributes to load (name, cls)
    :return: an instance of 'cls'
    """
    args = []

    for attribute_name, attribute_class in attributes:
        args.append(attribute_class.load(path, attribute_name))

    if os.path.exists(os.path.join(path, ".metadata")):
        with open(os.path.join(path, ".metadata")) as metadata_file:
            args.append(json.loads(metadata_file.read()))

    return cls(*args)


def _save_as_archive(obj, path, attributes, metadata=None):
    """Save the object as an uncompressed zip archive.

    :param obj: object to store
    :param path: the path to the archive where the files will be stored.
    :param attributes: list of attributes to save.
    :return: None
    """
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as zip_file:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for attribute in attributes:
                tmp_file = obj.__getattribute__(attribute).save(tmp_dir, name=attribute)
                zip_file.write(tmp_file, arcname=os.path.basename(tmp_file))

            if metadata:
                zip_file.writestr(".metadata", json.dumps(metadata, indent=4))


def _load_from_archive(cls, path, attributes):
    """Load an object from a zip archive file.

    :param cls: class of the object to load
    :param path: string, path to the archive where the object instance files are stored
    :param attributes: list of tuples defining attributes to load (name, cls)
    :return: an instance of 'cls'
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        with zipfile.ZipFile(path, 'r', zipfile.ZIP_STORED) as zip_file:
            zip_file.extractall(tmpdirname)
        return _load_from_folder(cls, tmpdirname, attributes)
