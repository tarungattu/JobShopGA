#number of machines
m = 4

class Machine:
    def __init__(self, machine_id):
        self.operationlist = []
        self.machine_id = machine_id % m #machine number
        self.lastJobCompTime = 0