from JobShopScheduler import JobShopScheduler
import benchmarks
import distances

def main():
    machine_data = benchmarks.pinedo['machine_data']
    ptime_data = benchmarks.pinedo['ptime_data']
    scheduler1 = JobShopScheduler(4, 3, 1, 50, 0.7, 0.5, 100, machine_data, ptime_data)    
    
    scheduler1.set_distance_matrix(distances.four_machine_matrix)
    scheduler1.display_schedule = 0
    
    scheduler2 = JobShopScheduler(4, 3, 3, 50, 0.7, 0.5, 100, machine_data, ptime_data)
    chromsome1 = scheduler1.GeneticAlgorithm()
    chromsome2 = scheduler2.GeneticAlgorithm()
    
    scheduler1.num_amrs = 2
    other_chromosome = scheduler1.GeneticAlgorithm()
    
    print('AMR MACHINE SEQUENCES')
    print(chromsome1.amr_machine_sequences)
    print(other_chromosome.amr_machine_sequences)
    
    # print(scheduler1.distance_matrix)
    
    
if __name__ == '__main__':
    main()