class Chromosome:
    def __init__(self, encoded_list):
        self.encoded_list = encoded_list
        self.ranked_list = []
        self.operation_index_list = []
        self.operation_schedule = []
        self.machine_sequence = []
        self.ptime_sequence = []
        self.Cmax = 9999
        self.penalty = 0
        self.fitness = self.Cmax + self.penalty