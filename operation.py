class Operation:
    def __init__(self, job_number):
        self.job_number = job_number
        self.operation_number = 0
        self.machine = None
        self.Pj = None
        self.start_time = 0
        self.Cj = 0
        self.travel_time = 0
        
    def getCj(self):
        self.Cj = self.start_time + self.Pj
        
    def calculate_travel_time(self, amrs, jobs, distance_matrix):
        distance = 0
        source = self.machine
        velocity = amrs[jobs[self.job_number].amr_number].velocity
        if self.Pj == 0:
            return 0
        
        # IF NEXT JOB PROCESSING TIME IS 0 PUT THIS JOB AS LAST JOB
        if self.operation_number != len(jobs[self.job_number].operations) - 1:
            if jobs[self.job_number].operations[self.operation_number + 1].Pj == 0:
                distance = 10
                # distance = 0
                return distance/velocity
                
        if self.operation_number == len(jobs[self.job_number].operations) - 1:
            distance = 10
            # distance = 0
        else:
            dest = jobs[self.job_number].operations[self.operation_number + 1].machine
            distance = distance_matrix[source][dest]
        return distance/velocity
            