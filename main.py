from JobShopScheduler import JobShopScheduler
import benchmarks
import distances

def main():
    machine_data = benchmarks.ft06['machine_data']
    ptime_data = benchmarks.ft06['ptime_data']
    scheduler1 = JobShopScheduler(6, 6, 2, 50, 0.7, 0.5, 100, machine_data, ptime_data)    
    
    scheduler1.set_distance_matrix(distances.six_machine_matrix)
    scheduler1.display_schedule = 1
    scheduler1.enable_travel_time = 1
    chromosome1 = scheduler1.GeneticAlgorithm()
    
    scheduler1.reschedule(2, 3)
    chromosome2 = scheduler1.GeneticAlgorithm()
    
    print('AMR MACHINE SEQUENCES')
    print(chromosome1.amr_machine_sequences)
    print(chromosome1.amr_ptime_sequences)
    print(chromosome2.amr_machine_sequences)
    print(chromosome2.amr_ptime_sequences)
    
    
if __name__ == '__main__':
    main()