# Johannes Siedersleben
# QAware GmbH, Munich
# 28.2.2021

from time import perf_counter

class Logger(object):
    def __init__(self):
        self.protocol = []
        self.counter = 0
        self.char_counter = 0

    def log(self, input: any) -> None:
        print(self.counter, end='')  # I am working
        self.counter = (self.counter + 1) % 10
        self.char_counter =(self.char_counter + 1) % 80
        if self.char_counter == 0:
          print()
        self.protocol.append((perf_counter(), input))

    def getProtocol(self) -> list:
        return self.protocol
