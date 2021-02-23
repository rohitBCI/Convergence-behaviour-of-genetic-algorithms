import numpy as np
import random
class Genetic_Algorithm():

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