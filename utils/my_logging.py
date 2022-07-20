import functools
import logging
import os
import sys


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, 'a')


def setup_logging(output_dir=None):
    _FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, format=_FORMAT, stream=sys.stdout)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s',
        datefmt='%m/%d %H:%M:%S',
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(plain_formatter)
    logger.addHandler(ch)

    if output_dir is None:
        return

    filename = os.path.join(output_dir, 'stdout.log')
    fh = logging.StreamHandler(_cached_log_stream(filename))
    fh.setLevel(logging.INFO)
    fh.setFormatter(plain_formatter)
    logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)
