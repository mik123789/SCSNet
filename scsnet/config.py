"""Immutable architecture parameters for SCSNet inference."""
from dataclasses import dataclass
from typing import Tuple


def _integer(name, value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError('{} must be an integer >= {}'.format(name, minimum))


def _triple(name, values):
    if not isinstance(values, tuple) or len(values) != 3:
        raise ValueError('{} must be a tuple of three positive integers'.format(name))
    for value in values:
        _integer(name, value)


@dataclass(frozen=True)
class SCSNetConfig:
    """Paper default: base width c=48 and OSCV group count G=48.

    The first search range is the number of nonnegative coarse candidates;
    subsequent ranges r describe signed candidates [-r+1, r-1].
    """

    c: int = 48
    G: int = 48
    stage_search_ranges: Tuple[int, int, int] = (24, 3, 3)
    sse_blocks: int = 2
    aggregation_layers: int = 6
    aggregation_channels: int = 16
    aggregation_growth: Tuple[int, int, int] = (6, 3, 3)
    ssr_channels: int = 32

    def __post_init__(self):
        for name in ('c', 'G', 'sse_blocks', 'aggregation_channels', 'ssr_channels'):
            _integer(name, getattr(self, name))
        _integer('aggregation_layers', self.aggregation_layers, minimum=0)
        _triple('stage_search_ranges', self.stage_search_ranges)
        _triple('aggregation_growth', self.aggregation_growth)
        if (2 * self.c) % self.G:
            raise ValueError('G must divide every ACFE output channel count (2*c, 4*c, 8*c)')
