from abc import ABC, abstractmethod

class BaseBuffer(ABC):
    @abstractmethod
    def store(self, **kwargs):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def get_data(self):
        pass
