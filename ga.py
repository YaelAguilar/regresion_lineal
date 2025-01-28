import random
import numpy as np
from genetic_strategy import GeneticStrategy
from objective import mean_squared_error

class BasicGeneticStrategy(GeneticStrategy):
    def __init__(self, function, X, Y, num_betas=4, bits_per_beta=16, crossover_prob=0.8, mutation_prob=0.1, bit_mutation_prob=0.01, elitism_rate=0.1):
        """
        Initialize the basic genetic strategy for linear regression.

        Parameters:
        - function: The objective function, here mean_squared_error.
        - X: numpy array of shape (n_samples, n_features)
        - Y: numpy array of shape (n_samples,)
        - num_betas: Number of betas (including intercept)
        - bits_per_beta: Number of bits to encode each beta
        - crossover_prob: Probability of crossover
        - mutation_prob: Probability of mutation per individual
        - bit_mutation_prob: Probability of mutation per bit
        - elitism_rate: Proportion of elite individuals to carry over
        """
        self.function = function
        self.X = X
        self.Y = Y
        self.num_betas = num_betas
        self.bits_per_beta = bits_per_beta
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.bit_mutation_prob = bit_mutation_prob
        self.elitism_rate = elitism_rate

    def setup_parameters(self, x_min, x_max, dx, min_population, max_population):
        """
        Setup parameters for the genetic algorithm.

        Parameters:
        - x_min: Minimum value for betas
        - x_max: Maximum value for betas
        - dx: Precision step (not used here)
        - min_population: Minimum population size
        - max_population: Maximum population size
        """
        self.x_min = x_min
        self.x_max = x_max
        self.dx = dx
        self.min_population = min_population
        self.max_population = max_population
        self.population_size = max_population

        # Total bits per individual
        self.total_bits = self.num_betas * self.bits_per_beta

        # System delta_x, used for mapping bits to real values
        self.dx_system = (self.x_max - self.x_min) / (2**self.bits_per_beta - 1) if self.bits_per_beta > 0 else self.dx

    def initialize_population(self):
        """
        Initialize the population with random binary strings representing betas.

        Returns:
        - population: list of binary strings
        """
        return [''.join(random.choice(['0', '1']) for _ in range(self.total_bits)) 
                for _ in range(self.population_size)]

    def fitness(self, individual):
        """
        Calculate the fitness of an individual based on MSE.

        Parameters:
        - individual: binary string

        Returns:
        - fitness: float
        """
        betas = self.decode_solution(individual)
        mse = self.function(betas, self.X, self.Y)
        # Fitness is inverse of mse
        fitness = 1 / (1 + mse)  # Add 1 to avoid division by zero
        return fitness

    def get_population_stats(self, population):
        """
        Get stats (betas and fitness) for the population.

        Parameters:
        - population: list of binary strings

        Returns:
        - betas_list: list of betas arrays
        - fitness_list: list of fitness values
        """
        betas_list = [self.decode_solution(ind) for ind in population]
        fitness_list = [self.fitness(ind) for ind in population]
        return betas_list, fitness_list

    def select_best(self, population):
        """
        Select the best individuals based on fitness.

        Parameters:
        - population: list of binary strings

        Returns:
        - selected: list of binary strings
        """
        # Evaluate fitness
        fitness_values = [(ind, self.fitness(ind)) for ind in population]
        # Sort by fitness descending
        fitness_sorted = sorted(fitness_values, key=lambda x: x[1], reverse=True)
        # Select top proportion based on elitism_rate
        n_selected = max(self.min_population, int(len(population) * self.elitism_rate))
        selected = [ind for ind, fit in fitness_sorted[:n_selected]]
        return selected

    def crossover(self, population):
        """
        Perform crossover on the population.

        Parameters:
        - population: list of binary strings

        Returns:
        - offspring: list of binary strings after crossover
        - num_pairs: number of crossover operations performed
        """
        offspring = []
        num_pairs = 0

        # Shuffle population
        shuffled = population.copy()
        random.shuffle(shuffled)

        # Pair individuals
        for i in range(0, len(shuffled)-1, 2):
            parent1 = shuffled[i]
            parent2 = shuffled[i+1]

            if random.random() <= self.crossover_prob:
                num_pairs +=1
                # Single-point crossover
                point = random.randint(1, self.total_bits -1)
                child1 = parent1[:point] + parent2[point:]
                child2 = parent2[:point] + parent1[point:]
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])

        if len(shuffled) %2 !=0:
            offspring.append(shuffled[-1])

        return offspring, num_pairs

    def mutate(self, population):
        """
        Perform mutation on the population.

        Parameters:
        - population: list of binary strings

        Returns:
        - mutated_population: list of binary strings after mutation
        - total_mutations: number of individuals mutated
        - total_mutated_bits: number of bits mutated
        """
        mutated_population = []
        total_mutations = 0
        total_mutated_bits = 0

        for individual in population:
            mutated_individual = list(individual)
            individual_mutated = False

            if random.random() <= self.mutation_prob:
                for i in range(self.total_bits):
                    if random.random() <= self.bit_mutation_prob:
                        mutated_individual[i] = '1' if mutated_individual[i] == '0' else '0'
                        total_mutated_bits +=1
                        individual_mutated = True

                if individual_mutated:
                    total_mutations +=1

            mutated_population.append(''.join(mutated_individual))

        return mutated_population, total_mutations, total_mutated_bits

    def get_best_and_worst(self, population):
        """
        Get the best and worst individuals based on fitness.

        Parameters:
        - population: list of binary strings

        Returns:
        - best_individual: binary string
        - worst_individual: binary string
        """
        fitness_values = [(ind, self.fitness(ind)) for ind in population]
        sorted_individuals = sorted(fitness_values, key=lambda x: x[1], reverse=True)
        return sorted_individuals[0][0], sorted_individuals[-1][0]

    def decode_solution(self, binary_string):
        """
        Decode the binary string into a list of betas.

        Parameters:
        - binary_string: string of '0's and '1's

        Returns:
        - betas: list of floats
        """
        betas = []
        for i in range(self.num_betas):
            start = i * self.bits_per_beta
            end = (i +1) * self.bits_per_beta
            gene = binary_string[start:end]
            decimal = int(gene, 2)
            # Map to real value
            real_val = self.x_min + decimal * self.dx_system
            betas.append(real_val)
        return betas

    def pruning(self, selected, offspring):
        """
        Combine selected and offspring populations, remove duplicates, and maintain population size.

        Parameters:
        - selected: list of binary strings
        - offspring: list of binary strings

        Returns:
        - new_population: list of binary strings
        """
        # Combine
        combined = selected + offspring

        # Remove duplicates
        unique = list(dict.fromkeys(combined))

        # Sort by fitness descending
        fitness_values = [self.fitness(ind) for ind in unique]
        sorted_inds = [ind for _, ind in sorted(zip(fitness_values, unique), key=lambda x: x[0], reverse=True)]

        # Maintain population size
        if len(sorted_inds) > self.max_population:
            new_population = sorted_inds[:self.max_population]
        else:
            new_population = sorted_inds

        return new_population
