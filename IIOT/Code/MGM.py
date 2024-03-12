from Agent import Agent
from Message import Message
from Message import Message_LR


class MGM(Agent):
    def __init__(self, id_agent):
        Agent.__init__(self, id_agent)
        LR = 0
        potential_val = 0


    def create_msg(self, to):
        msg = Message(self.iteration, self.id_agent, to,  self.value)
        return msg


    def create_LR_msg(self, to):
        msg = Message_LR(self.iteration, self.id_agent, to,  self.value, self.LR)
        return msg

    def add_neighbor(self, x):
        self.neighbors.append(x)

    def get_id(self):
        return self.id_agent

    def set_iteration(self, x):
        self.iteration = x

    def set_value(self, x):
        self.value = x

    def set_LR(self, x):
            self.LR = x

    def set_potential_val(self, x):
            self.potential_val = x