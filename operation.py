class Operation:
    def __init__(self, job_number, operation_number):
        self.job_number = job_number
        self.operation_number = operation_number
        self.machine = None
        self.Pj = None
        self.start_time = 0
        self.Cj = 0
        
    def getCj(self):
        self.Cj = self.start_time + self.Pj