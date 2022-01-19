from abc import ABC, abstractmethod
from numpy.random import default_rng
from coordinate import Coordinate
from visualizer_base import VisualizerBase


class ProblemBase(ABC):
    def __init__(self, **kwargs) -> None:
        self._random = default_rng(kwargs.get('seed', None))
        self._visualizer: VisualizerBase = None

    @abstractmethod
    def solve(self) -> Coordinate:
        pass

    def replay(self) -> None:
        """
        Start the problems visualization.
        """
        self._visualizer.replay()
