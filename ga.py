import numpy as np
import random

class LinearRegressionGA:
    def __init__(self, X, Y, learning_rate=0.01, generations=500, crossover_prob=0.8, mutation_prob=0.1, elitism_rate=0.1):
        self.X = X  # Matriz
        self.Y = Y  # Vector
        self.learning_rate = learning_rate
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism_rate = elitism_rate
        self.num_features = X.shape[1]
        self.population_size = 100
        self.history_mse = []
        self.history_fitness = []
        self.history_betas = []

    def initialize_population(self):
        return np.random.uniform(-10, 10, (self.population_size, self.num_features))

    def compute_fitness(self, population):
        # Fitness inverso al MSE
        predictions = population @ self.X.T
        mse = np.mean((predictions - self.Y) ** 2, axis=1)
        fitness = 1 / (mse + 1e-10)
        return fitness, mse

    def selection(self, population, fitness):
        # Selección por torneo
        selected = []
        for _ in range(self.population_size):
            i, j = random.sample(range(self.population_size), 2)
            if fitness[i] > fitness[j]:
                selected.append(population[i])
            else:
                selected.append(population[j])
        return np.array(selected)

    def crossover(self, parent1, parent2):
        # Cruce de un punto
        if random.random() < self.crossover_prob:
            point = random.randint(1, self.num_features - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        # Mutación gaussiana
        for i in range(self.num_features):
            if random.random() < self.mutation_prob:
                individual[i] += np.random.normal(scale=self.learning_rate)
        return individual

    def create_next_generation(self, selected_population):
        next_generation = []
        # Implementar elitismo
        fitness, _ = self.compute_fitness(selected_population)
        num_elites = max(1, int(self.elitism_rate * self.population_size))
        sorted_indices = np.argsort(-fitness)
        elites = selected_population[sorted_indices[:num_elites]]
        next_generation.extend(elites)
        
        # Generar el resto de la población
        while len(next_generation) < self.population_size:
            parent1, parent2 = random.sample(list(selected_population), 2)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            next_generation.append(child1)
            if len(next_generation) < self.population_size:
                next_generation.append(child2)
        return np.array(next_generation[:self.population_size])

    def run(self):
        population = self.initialize_population()
        for generation in range(self.generations):
            fitness, mse = self.compute_fitness(population)
            best_mse = np.min(mse)
            best_index = np.argmin(mse)
            best_individual = population[best_index]
            best_fitness = 1 / (best_mse + 1e-10)
            self.history_mse.append(best_mse)
            self.history_fitness.append(best_fitness)  # Almacenar el mejor fitness
            self.history_betas.append(best_individual.copy())
            # Selección
            selected = self.selection(population, fitness)
            # Crear próxima generación
            population = self.create_next_generation(selected)
            if (generation + 1) % 50 == 0:
                print(f"Generación {generation+1}/{self.generations} - Mejor MSE: {best_mse:.4f} - Mejor Fitness: {best_fitness:.4f}")
        fitness, mse = self.compute_fitness(population)
        best_mse = np.min(mse)
        best_index = np.argmin(mse)
        best_individual = population[best_index]
        return best_individual, best_mse
