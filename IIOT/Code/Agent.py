from abc import ABC, abstractmethod
import random


class Agent(ABC):

    @abstractmethod
    def __init__(self, id_agent):
        self.neighbors = []
        self.received_messages = []
        self.iteration = 0
        self.value = random.randint(0, 9)
        self.id_agent = id_agent

    @abstractmethod
    def add_neighbor(self, x):
        pass

    @abstractmethod
    def get_id(self, x):
        pass

    @abstractmethod
    def create_msg(self):
        pass

    @abstractmethod
    def set_iteration(self):
        pass

    @abstractmethod
    def set_value(self):
        pass
