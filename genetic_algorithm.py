import numpy as np
import random
class Genetic_Algorithm():

    def __init__(self, population, individuals, chromosome_length,scores,parent_1,parent_2, mutation_probability):
    	self.population = population
    	self.individuals= individuals
    	self.chromosome_length = chromosome_length
    	self.scores = scores
    	self.parent_1 = parent_1
    	self.parent_2 = parent_2s
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

    def calculate_fitness_individual(population):
        # Create an array of True/False compared to reference
        #identical_to_reference = population == reference
        # Sum number of genes that are identical to the reference
        fitness_scores_individual = np.sum(population)

        return fitness_scores_individual

    def select_individual_by_tournament(population, scores):
        # Get population size
        population_size = len(scores)

        # Pick individuals for tournament
        fighter_1 = random.randint(0, population_size-1)

        fighter_2 = random.randint(0, population_size-1)

        # Get fitness score for each
        fighter_1_fitness = scores[fighter_1]
        fighter_2_fitness = scores[fighter_2]

        # Identify undividual with highest fitness
        # Fighter 1 will win if score are equal
        if fighter_1_fitness >= fighter_2_fitness:
            winner = fighter_1
        else:
            winner = fighter_2

        # Return the chromsome of the winner
        return population[winner, :] 

    def breed_by_crossover(parent_1, parent_2):
        # Get length of chromosome
        chromosome_length = len(parent_1)

        # Pick crossover point, avoding ends of chromsome
        crossover_point = random.randint(1,chromosome_length-1)

        # Create children. np.hstack joins two arrays
        child_1 = np.hstack((parent_1[0:crossover_point],
                            parent_2[crossover_point:]))

        child_2 = np.hstack((parent_2[0:crossover_point],
                            parent_1[crossover_point:]))

        # Return children
        return child_1, child_2

    def randomly_mutate_population(population, mutation_probability):

        # Apply random mutation
        random_mutation_array = np.random.random(
            size=(population.shape))

        random_mutation_boolean = \
            random_mutation_array <= mutation_probability

        population[random_mutation_boolean] = \
        np.logical_not(population[random_mutation_boolean])

        # Return mutation population
        return population

    def randomly_mutate_population(population, mutation_probability):

        # Apply random mutation
        random_mutation_array = np.random.random(
            size=(population.shape))

        random_mutation_boolean = \
            random_mutation_array <= mutation_probability

        population[random_mutation_boolean] = \
        np.logical_not(population[random_mutation_boolean])

        # Return mutation population
        return population