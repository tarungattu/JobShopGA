class Operation:
    def __init__(self, job_number):
        self.job_number = job_number
        self.machine = None
        self.Pj = None
        self.start_time = 0
        self.Cj = 0