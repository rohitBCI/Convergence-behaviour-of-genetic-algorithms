# Import libraries
import numpy as np
import random
import time
import operator
from collections import defaultdict
from genetic_algorithm import ga
np.set_printoptions(suppress=True)

# Set general parameters
chromosome_length = 100
population_size = 10
trap = 1
k=4
d=2.5
tightly_linked = 0

run_dict = defaultdict(list)
reliable_num = 0

# Averaging stochastic effects of GA
while(reliable_num < 25):
    reliable_num += 1
    population_size_array = np.array([])
    outcome_array = np.array([]).reshape(0,2)
    new_population_size = 10
    crossover_operator = '2X'
    bisection_search = 0
    num_generation = 0 
    num_fitnessfunc = 0
    success_population_sizes = np.array([])
    
    while(True):
        start = time.time()
        # Create a random population
        population = ga.create_starting_population(int(new_population_size), chromosome_length)
        # Run the genetic algorithm
        outcome = ga.run_generation(population, crossover_operator,trap,k,d,tightly_linked)
        population_size_array = np.hstack((population_size_array,new_population_size))

        # Outcome Fail for the current population size
        if(outcome == "Fail" and new_population_size <= 2560):
            if(bisection_search == 0):
                # Population doubled for next iteration
                new_population_size = new_population_size*2
            else:
                # Perform bisection search to narrow down search space to optimal population size
                midpoint_gap = abs((population_size_array[-1] - population_size_array[-2]))/2
                if(midpoint_gap < 10):
                    break
                new_population_size =  population_size_array[-1] + midpoint_gap    

        else:
            success_population_sizes = np.hstack((success_population_sizes,new_population_size))
            outcome_array = np.vstack((outcome_array,outcome)) 
            # Base case where optimal solution is found at population size 10
            if(len(population_size_array)==1):
                break

            # Trigger bisection search     
            bisection_search = 1
            midpoint_gap = abs((population_size_array[-1] - population_size_array[-2]))/2
            
            if(midpoint_gap < 10):
                break
            new_population_size = population_size_array[-1] - midpoint_gap
        
    end = time.time()
    # Calculate time taken for running the GA
    elapsed_time = end - start
    # Retrieve optimal population size
    optimal_population_size = success_population_sizes[-1]
    
    # Add observations to dictionary
    if (str(int(reliable_num)) not in run_dict.keys()):
        run_dict[str(int(reliable_num))] = [0,0,0,0]    
    
    run_dict[str(int(reliable_num))][0] = optimal_population_size
    run_dict[str(int(reliable_num))][1] = outcome_array[-1][0]
    run_dict[str(int(reliable_num))][2] = outcome_array[-1][1]
    run_dict[str(int(reliable_num))][3] = elapsed_time
    

observations = np.sum(np.array(list(run_dict.values())),axis=0)
# Average observations over 25 runs
averaged_observations = observations/len(run_dict)

# Write observations to file
f = open("observations_new.txt", "a")

f.write(f'Chromosome_length: {chromosome_length}\n')
f.write(f'Crossover operator used: {crossover_operator}\n')
if(trap==0):
    f.write(f'Fitness function: Counting Ones\n')
else:
    if(d==1):
        if(tightly_linked==1):
            f.write(f'Deceptive (tightly-linked) (k={k},d={d})\n') 
        else:
            f.write(f'Deceptive (non-tightly-linked) (k={k},d={d})\n')
    else:
        if(tightly_linked==1):
            f.write(f'Non-Deceptive (tightly-linked) (k={k},d={d})\n') 
        else:
            f.write(f'Non-Deceptive (non-tightly-linked) (k={k},d={d})\n')
f.write(f'Optimal_population_size: {int(averaged_observations[0])}\n')
f.write(f'Average number of generations : {averaged_observations[1]} (std: {np.std(np.array(list(run_dict.values()))[:,1])})\n')
f.write(f'Average number of fitness function evaluations : {averaged_observations[2]} (std: {np.std(np.array(list(run_dict.values()))[:,2])})\n')
f.write(f'Average CPU time: {averaged_observations[3]} (std: {np.std(np.array(list(run_dict.values()))[:,3])})\n')
f.write(f'\n\n')
f.close()