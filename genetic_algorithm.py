import numpy as np
import random
import warnings

class ga():
    def __init__(self, population, individuals, chromosome_length,scores,parent_1,parent_2, mutation_probability):
    	self.population = population
    	self.individuals= individuals
    	self.chromosome_length = chromosome_length
    	self.scores = scores
    	self.parent_1 = parent_1
    	self.parent_2 = parent_2
    	self.mutation_probability = mutation_probability

    def create_starting_population(individuals, chromosome_length):
        # Set up an initial array of all zeros
        population = np.zeros((individuals, chromosome_length))
        # Loop through each row (individual)
        for i in range(individuals):
            # Choose a random number of ones to create
            ones = random.randint(0, chromosome_length)
            # Change the required number of zeros to ones
            population[i, 0:ones] = 1
            # Shuffle row
            np.random.shuffle(population[i])

        return population

    def calculate_fitness(population):
        # Create an array of True/False compared to reference
        #identical_to_reference = population == reference
        # Sum number of genes that are identical to the reference
        fitness_scores = [np.sum(p) for p in population]

        return fitness_scores


    def two_point_crossover(parent_1, parent_2):

        chromosome_length = len(parent_1)
        # Pick crossover points, avoding ends of chromsome
        crossover_point1 = random.randint(1,chromosome_length)
        crossover_point2 = random.randint(1,chromosome_length-1)

        if crossover_point2 >= crossover_point1:
            crossover_point2 += 1

        else:
            crossover_point1, crossover_point2 = crossover_point2, crossover_point1
        
        # Create children. np.hstack joins two arrays

        child_1 = np.hstack((parent_1[0:crossover_point1],
                            parent_2[crossover_point1:crossover_point2],
                            parent_1[crossover_point2:]))

        child_2 = np.hstack((parent_2[0:crossover_point1],
                            parent_1[crossover_point1:crossover_point2],
                            parent_2[crossover_point2:]))

        # Return children
        return child_1, child_2



    def uniform_crossover(parent_1, parent_2):
    	chromosome_length = len(parent_1)
    	child_1 = np.array([])
    	child_2 = np.array([])
    	for i in range(chromosome_length):
    		if(parent_1[i]==parent_2[i]):
    			child_1  = np.hstack((child_1,parent_1[i]))
    			child_2  = np.hstack((child_2,parent_1[i]))       
    		else:
    			random_sample1 = random.randint(0, 1)
    			child_1  = np.hstack((child_1,random_sample1))
    			random_sample2 = random.randint(0, 1)
    			child_2  = np.hstack((child_2,random_sample2))
    	return child_1, child_2


    def check_stopping_criterion(fitness_values):
    	parent1_fitness = fitness_values[0]
    	parent2_fitness = fitness_values[1]
    	child1_fitness = fitness_values[2]
    	child2_fitness = fitness_values[3]

    	if(parent1_fitness > parent2_fitness):
    		max_parent_fitness = parent1_fitness

    	else:
    		max_parent_fitness = parent2_fitness

    	if(child1_fitness <= max_parent_fitness and child2_fitness <= max_parent_fitness):
    		return True

    	else:
    		return False

    def run_generation(population, crossover_operator):
    	num_generation = 0
    	best_score_progress = []
    	while(True):
		    num_generation += 1
		    flag = 1
		    population_size = len(population) 
		    chromosome_length = len(population[0])
		    # Create an empty list for new population
		    new_population = np.array([]).reshape(0, chromosome_length)
		    # Create new popualtion generating two children at a time
		    for i in range(0,population_size-1,2):
		        parent_1 = population[i]
		        parent_2 = population[i+1]
		        if crossover_operator == 'UX':
		        	child_1, child_2 = ga.uniform_crossover(parent_1, parent_2)
		        elif crossover_operator == '2X':
		        	child_1, child_2 = ga.two_point_crossover(parent_1, parent_2)
		       	else:
		       		warnings.warn(
         "Use '2X' for two-point crossover and 'UX' for uniform crossover")
 
		        family = np.array([parent_1,parent_2,child_1,child_2])
		        family_fitness = ga.calculate_fitness(family)
		        #print(family_fitness,flag)
		        stopping_criterion  = ga.check_stopping_criterion(family_fitness)
		        if(stopping_criterion == False):
		            flag=0
		        best_two = np.argsort(family_fitness)[::-1][:2]
		        best_in_family = family[best_two]
		        new_population =  np.vstack((new_population, best_in_family))
		        
		    # Replace the old population with the new one
		    population = np.array(new_population)
		    
		    # Score best solution, and add to tracker
		    scores = ga.calculate_fitness(population)
		    best_score = np.max(scores)/chromosome_length * 100
		    best_score_progress.append(best_score)
		    if(best_score==100):
		        return (population[np.argmax(scores)])
		    if(flag==1):
		        return "Fail"
		#     else:
		#         print(f'Generation : {num_generation}, score : {best_score}')
		# GA has completed required generation
		#print ('End best score, percent target: %.1f' %best_score)
		
		    
		    