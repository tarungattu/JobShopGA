import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from job import Job
from machine import Machine
from operation import Operation
from chromosome import Chromosome

m = 4
n = 3
N = 2
pc = 0.5


machine_data = [0,1,2,3, 1,0,3,2, 0,1,3,2]
ptime_data = [10,8,4,0, 4,3,5,6, 4,7,3,0]


# print out necessary
if len(sys.argv) > 1:
    print_out = sys.argv[1].lower() == 'true'
else:
    # Default value if no command-line argument is provided
    print_out = False
    
if len(sys.argv) > 1:
    test_old_prog = sys.argv[1].lower() == 'old'
else:
    # Default value if no command-line argument is provided
    test_old_prog = False

if len(sys.argv) > 1:
    processing = sys.argv[1].lower() == 'proc'
else:
    # Default value if no command-line argument is provided
    processing = False

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
    
    # merged_array = np.array([machine_data, ptime_data])

    # # Reshape the array to get the desired format
    # reshaped_array = merged_array.reshape((len(machine_data) // len(set(machine_data)), len(set(machine_data)), 2))
    # return reshaped_array

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

def induv_integer_list(chromosome):
    ranked_population = []
    sorted_list = []
    ranks = {}
    # Sort the list to get ranks in ascending order
    sorted_list = sorted(chromosome)
            
    # Create a dictionary to store the ranks of each float number
    ranks = {value: index for index, value in enumerate(sorted_list)}
            
    # Convert each float number to its corresponding rank
    rank_list = [ranks[value] + 1 for value in chromosome]
    ranked_population.append(rank_list)
        
    return rank_list
    
    
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

def induv_getJobindex(chromosome):
    new_index = 0
    operation_index_pop = []

    tlist = []
    temp = chromosome
    for j in range(len(chromosome)):
        new_index = (temp[j] % n) + 1
        tlist.append(new_index)
    operation_index_pop = tlist
    
    return operation_index_pop
    

def schedule_operations(population, jobs):
    operation_list = []
    explored = []
    
    for chromosome in population:
        for i in range(len(chromosome) - 1):
            explored.append(chromosome[i])
            numcount = explored.count(chromosome[i])
            if numcount <= m:
                operation_list.append(jobs[chromosome[i]-1].operations[numcount-1])
    return operation_list

def induv_schedule_operations(chromosome, jobs):
    operation_list = []
    explored = []
    
    for i in range(len(chromosome)):
        explored.append(chromosome[i])
        numcount = explored.count(chromosome[i])
        if numcount <= m:
            operation_list.append(jobs[chromosome[i]-1].operations[numcount-1])
    return operation_list
            
# gives each operation a job number of whihc job it is part of
def install_operations(jobs):
    for job in jobs:
        job.operations = [Operation(job.job_number) for i in range(m)]

operation_data = create_operation_data(machine_data,ptime_data, m)

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
            # print(f'machine no: {machines[operation.machine].machine_id}, new finish time :{machines[operation.machine].finish_operation_time}')
            
        else:
            if jobs[operation.job_number].operations[operation.operation_number - 1].Cj < machines[operation.machine].  finish_operation_time:
                operation.start_time = machines[operation.machine].finish_operation_time
                operation.Cj = operation.start_time + operation.Pj
                machines[operation.machine].finish_operation_time = operation.Cj
                # print(f'machine no: {machines[operation.machine].machine_id}, new finish time :{machines[operation.machine].finish_operation_time}')
                
            else:
                operation.start_time = jobs[operation.job_number].operations[operation.operation_number - 1].Cj
                operation.Cj = operation.start_time + operation.Pj
                if operation.Pj != 0:
                    machines[operation.machine].finish_operation_time = operation.Cj
                # print(f'machine no: {machines[operation.machine].machine_id}, new finish time :{machines[operation.machine].finish_operation_time}')
                

def assign_machine_operationlist(machines, operation_schedule):
    for operation in operation_schedule:
        machines[operation.machine].operationlist.append(operation)

def get_Cmax(machines):
    runtimes = []
    for machine in machines:
        runtimes.append(machine.finish_operation_time)
        
    return max(runtimes)

def process_chromosome(chromosome):
    operation_data = create_operation_data(machine_data,ptime_data, m)
    print(operation_data)
    jobs = [Job(number) for number in range(n)]
    machines = [Machine(number) for number in range(m)]
    assign_operations(jobs, operation_data)
    
    ranked_list = induv_integer_list(chromosome)
    operation_index_list = induv_getJobindex(ranked_list)
    
    # CASE 1
    # operation_index_list = [2, 0, 2, 1, 0, 2, 0, 1, 1, 1, 2, 0]
    
    
    install_operations(jobs)
    assign_data_to_operations(jobs, operation_data)
    operation_schedule = induv_schedule_operations(operation_index_list, jobs)
    
    # get the sequence of machines
    machine_sequence = get_machine_sequence(operation_schedule)
    
    # get the sequence of processing times
    ptime_sequence = get_processing_times(operation_schedule)
    
    calculate_Cj(operation_schedule, machines, jobs, machine_sequence, ptime_sequence)
    assign_machine_operationlist(machines, operation_schedule)
    Cmax = get_Cmax(machines)
    
    chromosome = Chromosome(chromosome)
        
    chromosome.ranked_list = ranked_list
    chromosome.operation_index_list = operation_index_list
    chromosome.operation_schedule = operation_schedule
    chromosome.machine_sequence = machine_sequence
    chromosome.machine_list = machines
    chromosome.ptime_sequence = ptime_sequence
    chromosome.Cmax = Cmax
    
    return chromosome

def PlotGanttChar (chromosome):
        # ------------------------------
        # Figure and set of subplots
        
    Cmax = chromosome.Cmax
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(10)
    # ylim and xlim of the axes
    ax.set_ylabel('Machine', fontweight ='bold', loc='top', color='magenta', fontsize=16)
    ax.set_ylim(-0.5, m-0.5)
    ax.set_yticks(range(m), minor=False)
    ax.tick_params(axis='y', labelcolor='magenta', labelsize=16)
        
    ax.set_xlabel('Time', fontweight ='bold', loc='right', color='red', fontsize=16)
    ax.set_xlim(0, Cmax+2)
        
    ax.tick_params(axis='x', labelcolor='red', labelsize=16)
        
    ax.grid(True)
        
    tmpTitle = 'Job Shop Scheduling (m={:02d}; n={:03d}; Utilization={:04d})'.format(m, n, Cmax)
    plt.title(tmpTitle, size=24, color='blue')
        
    colors = ['orange', 'deepskyblue', 'indianred', 'limegreen', 'slateblue', 'gold', 'violet', 'grey', 'red', 'magenta','blue','green','silver']
        
        
    for i in range (m):
        joblen = len(chromosome.machine_list[i].operationlist)
        for k in range(joblen):
            j = chromosome.machine_list[i].operationlist[k]
            ST = j.start_time
            cIndx = 0
            # cIndx = k%(n*N)
            if j.Pj != 0:
                ax.broken_barh([(ST, j.Pj)], (-0.3+i, 0.6), facecolor=colors[j.job_number], linewidth=1, edgecolor='black')
                ax.text((ST + (j.job_number/2-0.3)), (i+0.03), '{}'.format(j.job_number), fontsize=18)
                
def tournament(population):
    indices2 = [x for x in range(N)]
    
    winners = []
    while len(indices2) != 0:
        i1 = random.choice(indices2)
        i2 = random.choice(indices2)
        while i1 == i2:
            i2 = random.choice(indices2)
            
        if population[i1].fitness < population[i2].fitness:
            winners.append(population[i1])
        else:
            winners.append(population[i2])
            
        indices2.remove(i1)
        indices2.remove(i2)
    
    indices2 = [x for x in range(N)]
    
    winners = []
    while len(indices2) != 0:
        i1 = random.choice(indices2)
        i2 = random.choice(indices2)
        while i1 == i2:
            i2 = random.choice(indices2)
            
        if population[i1].fitness < population[i2].fitness:
            winners.append(population[i1])
        else:
            winners.append(population[i2])
            
        indices2.remove(i1)
        indices2.remove(i2)
        
    return winners
    
    
                
def single_point_crossover(chrom1, chrom2):
    
    parent1 = chrom1.encoded_list
    parent2 = chrom2.encoded_list
    
    # r = random.uniform(0,1)
    r = 0.4
    
    p = 5
    if r > pc:
        return chrom1 , chrom2
    else:
        offspring1 = parent1[0:p] + parent2[p:]
        offspring2 = parent2[0:p] + parent1[p:]
        chrom_out1 = process_chromosome(offspring1)
        chrom_out2 = process_chromosome(offspring2)
    
    return chrom_out1, chrom_out2
    
def swapping(chromosome):
    code = chromosome.encoded_list
    indexes = [num for num in range(len(code))]
    
    p = random.choice(indexes)
    q = random.choice(indexes)
    while p == q:
        q = random.choice(indexes)
        
    print(code)
        
    code[p], code[q] = code[q], code[p]
    print(code)

def inversion(chromosome):
    code = chromosome.encoded_list
    indexes = [num for num in range(len(code))]
    p = random.choice(indexes)
    q = random.choice(indexes)
    while p == q:
        q = random.choice(indexes)
        
    print(code)
    p, q = min(p, q), max(p, q)
    code[p:q+1] = reversed(code[p:q+1])
    print(code)

def main1():
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
        
def main2():
    initial_population = generate_population(N)
    
    population = []
    # for encoded_list in initial_population:
    #     print(f'generated list: {encoded_list}')
    #     chromosome = process_chromosome(encoded_list)
    #     population.append(chromosome)
    
    encoded_list1 = [7.45,	10.69,	9.73,	1.31,	1.67,	1.58,	7.29,	2.77,	8.91,	7.35,	3.46,	7.47]
    chromosome_test1 = process_chromosome(encoded_list1)
    
    population.append(chromosome_test1)
    
    encoded_list2 = [4.74, 8.05, 10.48, 7.19, 6.05, 0.56, 0.04, 3.82, 1.37, 3.95, 1.46, 5.38]
    chromosome_test2 = process_chromosome(encoded_list2)
    
    population.append(chromosome_test2)
    
    
    
    
    # for chromosome in population:
    #     for machine in chromosome.machine_list:
    #         for operation in machine.operationlist:
    #             print(f'machine no: {machine.machine_id}, operation assigned mach: {operation.machine}, job no: {operation.job_number}, operation no: {operation.operation_number}')
            
        
        
    # PlotGanttChar(chromosome_test)
    # plt.show()
    
    # winners_list = tournament(population)
    
    # print('parents are')
    # for chromosome in winners_list:
    #     print(chromosome.encoded_list)
    
    # serial crossover section
    
    # indices = [x for x in range(N)]
    # offspring_list = winners_list
    # while len(indices) != 0:
    #     i1 = random.choice(indices)
    #     i2 = random.choice(indices)
    #     while i1 == i2:
    #         i2 = random.choice(indices)
        
    #     offspring1, offspring2 = single_point_crossover(winners_list[i1], winners_list[i2])
    #     offspring_list[i1] = offspring1
    #     offspring_list[i2] = offspring2
        
    #     indices.remove(i1)
    #     indices.remove(i2)
    
    offspring_list = []
    
    offspring1, offspring2 = single_point_crossover(chromosome_test1, chromosome_test2)
    offspring_list.extend([offspring1, offspring2])
    
    # swapping(chromosome_test1)
    
    inversion(chromosome_test1)
    
        
    if print_out:
        for chromosome in population:
            print('random generated numbers:',chromosome.encoded_list)
            print(f'ranked list : {chromosome.ranked_list}\n operation_index :{chromosome.operation_index_list},\n operation object{chromosome.operation_schedule}\n')
            print(f'machine sequence: {chromosome.machine_sequence}\n ptime sequence: {chromosome.ptime_sequence}\n Cmax: {chromosome.Cmax}')
            for machine in chromosome.machine_list:
                print(f'machine no: {machine.machine_id}, Cj :{machine.finish_operation_time}')
        for chromosome in offspring_list:
            print('random generated numbers:',chromosome.encoded_list)
            print(f'ranked list : {chromosome.ranked_list}\n operation_index :{chromosome.operation_index_list},\n operation object{chromosome.operation_schedule}\n')
            print(f'machine sequence: {chromosome.machine_sequence}\n ptime sequence: {chromosome.ptime_sequence}\n Cmax: {chromosome.Cmax}')
            for machine in chromosome.machine_list:
                print(f'machine no: {machine.machine_id}, Cj :{machine.finish_operation_time}')
                
    # print('offsprings are:')
    # for chromosome in offspring_list:
    #     print(chromosome.encoded_list)
    
        
if __name__ == '__main__':
    main2()