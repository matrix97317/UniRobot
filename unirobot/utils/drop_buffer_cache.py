# -*- coding: utf-8 -*-
"""Drop buffer cache."""
import logging
import os


logger = logging.getLogger(__name__)


def _drop_buffer_cache(filepath: str) -> None:
    """Drop buffer cache for a specific file path."""
    file_descriptor = os.open(filepath, os.O_RDONLY)
    try:
        os.posix_fadvise(file_descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    except Exception as ex:
        raise RuntimeError("Drop buffer cache failed.") from ex
    logger.info("Drop cache for `%s` success.", filepath)
    os.close(file_descriptor)


def drop_buffer_cache(target_path: str) -> None:
    """Drop buffer cache for a file or a directory."""
    if os.path.islink(target_path):
        target_path = os.readlink(target_path)
    if not os.path.exists(target_path):
        raise ValueError(f"The path `{target_path}` does not exist.")
    if os.path.isfile(target_path):
        logger.info("Running drop_buffer_cache for file `%s`.", target_path)
        _drop_buffer_cache(target_path)
    else:
        logger.info("Running drop_buffer_cache for dir `%s`.", target_path)
        for dirpath, _, filenames in os.walk(target_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                _drop_buffer_cache(filepath)
