# -*- coding: utf-8 -*-

import os


def root_dir():
    """root_dir"""
    """ first use realpath for symbolic path,
        then use abspath for relative path
    """
    return os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
