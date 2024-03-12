

class MailBox:

    def __init__(self):
        self.messages = []

    def add_message(self, x):
        self.messages.append(x)

    def remove_message(self, x):
        self.messages.remove(x)

