import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from job import Job
from machine import Machine
from operation import Operation

m = 4
n = 3
N = 1


machine_data = [0,1,2,3, 1,0,3,2, 0,1,3,2]
ptime_data = [10,8,4,0, 4,3,5,6, 4,7,3,0]



if len(sys.argv) > 1:
    print_out = sys.argv[1].lower() == 'true'
else:
    # Default value if no command-line argument is provided
    print_out = False
    


def create_operation_data(machine_data, ptime_data, m):
    matrix = []
    sublist = []
    for i in range(len(machine_data)):
        sublist.append([machine_data[i], ptime_data[i]])
        if (i + 1) % m == 0:
            matrix.append(sublist)
            sublist = []
    # Check if there are remaining elements
    if sublist:
        matrix.append(sublist)
    return matrix

def assign_operations(jobs, operation_data):
    for job, operation in zip(jobs, operation_data):
        job.operations = operation
        
        
def generate_population(N):
    population = []
    for _ in range(N):
        num = [round(random.uniform(0,m*n), 2) for _ in range(n*m)]
        population.append(num)
    return population

def integer_list(population):
    population[0] = [7.45,	10.69,	9.73,	1.31,	1.67,	1.58,	7.29,	2.77,	8.91,	7.35,	3.46,	7.47]
    ranked_population = []
    for i in range(N):
        sorted_list = []
        ranks = {}
        # Sort the list to get ranks in ascending order
        sorted_list = sorted(population[i])
            
        # Create a dictionary to store the ranks of each float number
        ranks = {value: index + 1 for index, value in enumerate(sorted_list)}
            
        # Convert each float number to its corresponding rank
        rank_list = [ranks[value] for value in population[i]]
        ranked_population.append(rank_list)
        
    return ranked_population

# get job operation sequence
def getJobindex(population):
    new_index = 0
    operation_index_pop = []
    for i in range(N):
        tlist = []
        temp = population[i]
        for j in range(m*n):
            new_index = (temp[j] % n) + 1
            tlist.append(new_index)
        operation_index_pop.append(tlist)
    
    return operation_index_pop

def schedule_operations(population, jobs):
    operation_list = []
    explored = []
    
    for chromosome in population:
        for i in range(len(chromosome)):
            explored.append(chromosome[i])
            numcount = explored.count(chromosome[i])
            if numcount <= m:
                operation_list.append(jobs[chromosome[i] - 1].operations[numcount - 1])   # possible bug: difference in answer when input is given directly.

    return operation_list

# gives each operation a job number of whihc job it is part of
def install_operations(jobs):
    for job in jobs:
        job.operations = [Operation(job.job_number) for i in range(m)]
        
def assign_data_to_operations(jobs, operation_data):
    for job,sublist in zip(jobs, operation_data):
        for operation,i in zip(job.operations, range(m)):
            operation.operation_number = i
            operation.machine = sublist[i][0]
            operation.Pj = sublist[i][1]
            
def get_machine_sequence(operation_schedule):
    machine_sequence = []
    for operation in operation_schedule:
        machine_sequence.append(operation.machine)
        
    return machine_sequence

def get_processing_times(operation_schedule):
    ptime_sequence = []
    for operation in operation_schedule:
        ptime_sequence.append(operation.Pj)
        
    return ptime_sequence
            

def calculate_Cj(operation_schedule, machines, jobs, machine_sequence, ptime_sequence):
    for operation in operation_schedule:
        if operation.operation_number == 0:
            operation.start_time = machines[operation.machine].finish_operation_time
            operation.Cj = operation.start_time + operation.Pj
            machines[operation.machine].finish_operation_time = operation.Cj
            print(f'machine no: {machines[operation.machine].machine_id}, new finish time :{machines[operation.machine].finish_operation_time}')
            
        else:
            if jobs[operation.job_number].operations[operation.operation_number - 1].Cj < machines[operation.machine].  finish_operation_time:
                operation.start_time = machines[operation.machine].finish_operation_time
                operation.Cj = operation.start_time + operation.Pj
                machines[operation.machine].finish_operation_time = operation.Cj
                print(f'machine no: {machines[operation.machine].machine_id}, new finish time :{machines[operation.machine].finish_operation_time}')
                
            else:
                operation.start_time = jobs[operation.job_number].operations[operation.operation_number - 1].Cj
                operation.Cj = operation.start_time + operation.Pj
                if operation.Pj != 0:
                    machines[operation.machine].finish_operation_time = operation.Cj
                print(f'machine no: {machines[operation.machine].machine_id}, new finish time :{machines[operation.machine].finish_operation_time}')
                
def get_Cmax(machines):
    runtimes = []
    for machine in machines:
        runtimes.append(machine.finish_operation_time)
        
    return max(runtimes)

operation_data = create_operation_data(machine_data,ptime_data, m)


jobs = [Job(number) for number in range(n)]
machines = [Machine(number) for number in range(m)]


assign_operations(jobs, operation_data)

initial_population = generate_population(N)
ranked_population = integer_list(initial_population)
operation_index_pop = getJobindex(ranked_population)

# CASE 1
# operation_index_pop = [[2, 0, 2, 1, 0, 2, 0, 1, 1, 1, 2, 0]]

# CASE 2
# operation_index_pop = [[0, 1, 2, 1, 0, 0, 2, 2, 1, 1, 0, 2]] 

# install the operations in each job
install_operations(jobs)
assign_data_to_operations(jobs, operation_data)
# create sequence with actual operations
operation_schedule = schedule_operations(operation_index_pop, jobs)

# get the sequence of machines
machine_sequence = get_machine_sequence(operation_schedule)

# get the sequence of processing times
ptime_sequence = get_processing_times(operation_schedule)

calculate_Cj(operation_schedule, machines, jobs, machine_sequence, ptime_sequence)
Cmax = get_Cmax(machines)

if print_out:
    print(operation_data)
    print('Job 0 operations', jobs[0].operations[0].job_number)
    print('Job 1 operations', jobs[1].operations[1].job_number)
    print('Job 2 operations',jobs[2].operations)
    print('initial population: \n', initial_population)
    print('ranked list:\n', ranked_population)
    print('job operation sequence list:\n', operation_index_pop)
    print('job operation sequence:\n', operation_schedule)
    print(f'machine sequence: {machine_sequence}')
    print(f'ptime sequence: {ptime_sequence}')
    
    for operation in operation_schedule:
        print(f'\n operation of job number: {operation.job_number},operation number: {operation.operation_number}, operation machine number :{ operation.machine}, processing time:{operation.Pj}\n Start time: {operation.start_time}, Pj: {operation.Pj }, Cj: {operation.Cj}')
        
    for machine in machines:
        print(f'machine number: {machine.machine_id}, machine finish: {machine.finish_operation_time}')
        
    print(f'Cmax is {Cmax}')