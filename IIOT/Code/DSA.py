from Agent import Agent
from Message import Message


class DSA(Agent):
    def __init__(self, id_agent):
        Agent.__init__(self, id_agent)

    def create_msg(self, to):
        msg = Message(self.iteration, self.id_agent, to,  self.value)
        return msg

    def add_neighbor(self, x):
        self.neighbors.append(x)

    def get_id(self):
        return self.id_agent

    def set_iteration(self, x):
        self.iteration = x

    def set_value(self, x):
        self.value = x
