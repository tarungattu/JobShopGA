#number of machines
m = 4

class Machine:
    def __init__(self, machine_id):
        self.joblist = []
        self.machine_id = machine_id % m #machine number
        self.start_operation_time = 0
        self.finish_operation_time = 0
        