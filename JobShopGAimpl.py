import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from job import Job
from machine import Machine

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
    ranked_population = []
    for i in range(N):
        sorted_list = []
        ranks = {}
        # Sort the list to get ranks in ascending order
        sorted_list = sorted(population[i])
            
        # Create a dictionary to store the ranks of each float number
        ranks = {value: index for index, value in enumerate(sorted_list)}
            
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
            operation_list.append(jobs[chromosome[i]-1].operations[numcount-1])
    return operation_list
            

operation_data = create_operation_data(machine_data,ptime_data, m)


jobs = [Job(number) for number in range(n)]

assign_operations(jobs, operation_data)

initial_population = generate_population(N)
ranked_population = integer_list(initial_population)
operation_index_pop = getJobindex(ranked_population)
operation_schedule = schedule_operations(operation_index_pop, jobs)

if print_out:
    print(operation_data)
    print('Job 1 operations', jobs[0].operations)
    print('Job 2 operations', jobs[1].operations)
    print('Job 3 operations',jobs[2].operations)
    print('initial population: \n', initial_population)
    print('ranked list:\n', ranked_population)
    print('job operation sequence list:\n', operation_index_pop)
    print('operation schedule:\n', operation_schedule)