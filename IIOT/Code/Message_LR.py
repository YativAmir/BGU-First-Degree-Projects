class Message_LR():
    def __init__(self, iteration, from_id, to_id, value_from, lr):
        self.iteration = iteration
        self.from_id = from_id
        self.to_id = to_id
        self.value_from = value_from
        self.lr = lr