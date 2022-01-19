from abc import ABC, abstractmethod

class VisualizerBase(ABC):
    @abstractmethod
    def add_data(self) -> None:
        pass

    @abstractmethod
    def replay(self, **kwargs)-> None:
        pass
