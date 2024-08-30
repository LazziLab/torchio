"""Top-level package for torchio."""

__author__ = """Fernando Perez-Garcia"""
__email__ = 'fepegar@gmail.com'
__version__ = '0.19.10'
# modified by Andres Morales for lazy operations

from . import utils
from .constants import *  # noqa: F401, F403
from .transforms import *  # noqa: F401, F403
from .data import (
    io,
    sampler,
    inference,
    SubjectsDataset,
    Image,
    ScalarImage,
    LabelMap,
    Queue,
    Subject,
    WeightedSampler,
    UniformSampler,
    LabelSampler,
    GridSampler,
    GridAggregator,
    Lazy_SubjectsDataset,
    Lazy_Image,
    Lazy_ScalarImage,
    Lazy_LabelMap,
    Lazy_Subject,
    Lazy_WeightedSampler,
    Lazy_UniformSampler,
    Lazy_LabelSampler,
    Lazy_GridSampler,
)
from . import datasets
from . import reference


__all__ = [
    'utils',
    'io',
    'sampler',
    'inference',
    'SubjectsDataset',
    'Image',
    'ScalarImage',
    'LabelMap',
    'Queue',
    'Subject',
    'datasets',
    'reference',
    'WeightedSampler',
    'UniformSampler',
    'LabelSampler',
    'GridSampler',
    'GridAggregator',
    'Lazy_SubjectsDataset',
    'Lazy_Image',
    'Lazy_ScalarImage',
    'Lazy_LabelMap',
    'Lazy_Subject',
    'Lazy_WeightedSampler',
    'Lazy_UniformSampler',
    'Lazy_LabelSampler',
    'Lazy_GridSampler',
]
