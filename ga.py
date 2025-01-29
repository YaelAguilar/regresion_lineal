# ga.py
import math
import random
import numpy as np

class LinearRegressionGA:
    def __init__(self, X, Y, generations=500, crossover_prob=0.8, mutation_prob=0.1, 
                 bit_mutation_prob=0.1, initial_population_size=50, max_population_size=100, elitism_rate=0.1):
        """
        Inicializa el Algoritmo Genético para regresión lineal.
        
        Parámetros:
            X (np.ndarray): Matriz de características normalizadas con intercepto.
            Y (np.ndarray): Vector de variables dependientes normalizadas.
            generations (int): Número de generaciones a ejecutar.
            crossover_prob (float): Probabilidad de realizar cruce entre padres.
            mutation_prob (float): Probabilidad de mutar un individuo.
            bit_mutation_prob (float): Probabilidad de mutar cada gen dentro de un individuo.
            initial_population_size (int): Tamaño de la población inicial.
            max_population_size (int): Tamaño máximo permitido para la población.
            elitism_rate (float): Proporción de individuos élite a mantener en la próxima generación.
        """
        self.X = X  # Matriz de características con intercepto
        self.Y = Y  # Vector de variables dependientes normalizado
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.bit_mutation_prob = bit_mutation_prob
        self.initial_population_size = initial_population_size
        self.max_population_size = max_population_size
        self.elitism_rate = elitism_rate
        self.num_features = X.shape[1]
        self.history_mse = []
        self.history_fitness = []
        self.history_betas = []

        # Validaciones de parámetros
        if self.initial_population_size > self.max_population_size:
            raise ValueError("El tamaño inicial de la población no puede exceder el tamaño máximo de la población.")
        if not (0 <= self.crossover_prob <=1):
            raise ValueError("La probabilidad de cruce debe estar entre 0 y 1.")
        if not (0 <= self.mutation_prob <=1):
            raise ValueError("La probabilidad de mutación debe estar entre 0 y 1.")
        if not (0 <= self.bit_mutation_prob <=1):
            raise ValueError("La probabilidad de mutación del gen debe estar entre 0 y 1.")
        if not (0 <= self.elitism_rate <=1):
            raise ValueError("La tasa de elitismo debe estar entre 0 y 1.")

    def initialize_population(self):
        """
        Inicializa la población con individuos aleatorios.
        
        Retorna:
            np.ndarray: Población inicial.
        """
        # Generar población inicial
        population = np.random.uniform(-10, 10, (self.initial_population_size, self.num_features))
        
        # Si el tamaño inicial es menor que el máximo, llenar el resto con individuos aleatorios
        if self.max_population_size > self.initial_population_size:
            remaining = self.max_population_size - self.initial_population_size
            additional_population = np.random.uniform(-10, 10, (remaining, self.num_features))
            population = np.vstack((population, additional_population))
        
        return population

    def compute_fitness(self, population):
        """
        Calcula la aptitud de cada individuo en la población.
        
        Retorna:
            tuple: (fitness array, mse array)
        """
        predictions = population @ self.X.T
        mse = np.mean((predictions - self.Y) ** 2, axis=1)
        fitness = 1 / (mse + 1e-10)  # Añadir epsilon para evitar división por cero
        return fitness, mse

    def selection(self, population, fitness):
        """
        Selección por torneo: selecciona mejores individuos para la reproducción.
        
        Retorna:
            np.ndarray: Individuos seleccionados.
        """
        selected = []
        for _ in range(self.max_population_size):
            i, j = random.sample(range(self.max_population_size), 2)
            if fitness[i] > fitness[j]:
                selected.append(population[i])
            else:
                selected.append(population[j])
        return np.array(selected)

    def crossover(self, parent1, parent2):
        """
        Realiza el cruce de dos padres para producir dos hijos.
        
        Retorna:
            tuple: (hijo1, hijo2)
        """
        if random.random() < self.crossover_prob:
            # Cruce de un punto
            if self.num_features > 1:
                point = random.randint(1, self.num_features - 1)
            else:
                point = 1
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        """
        Aplica mutación gaussiana a un individuo basado en las probabilidades de mutación.
        
        Retorna:
            np.ndarray: Individuo mutado.
        """
        # Decidir si mutar el individuo completo
        if random.random() < self.mutation_prob:
            for i in range(self.num_features):
                if random.random() < self.bit_mutation_prob:
                    # Añadir ruido gaussiano con una desviación estándar fija (e.g., 0.1)
                    individual[i] += np.random.normal(scale=0.1)
        return individual

    def create_next_generation(self, selected_population):
        """
        Crea la próxima generación aplicando elitismo, cruce y mutación.
        
        Retorna:
            np.ndarray: Nueva población.
        """
        next_generation = []
        # Implementar elitismo
        fitness, _ = self.compute_fitness(selected_population)
        num_elites = max(1, int(self.elitism_rate * self.max_population_size))
        sorted_indices = np.argsort(-fitness)
        elites = selected_population[sorted_indices[:num_elites]]
        next_generation.extend(elites)
        
        # Generar el resto de la población
        while len(next_generation) < self.max_population_size:
            parent1, parent2 = random.sample(list(selected_population), 2)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            next_generation.append(child1)
            if len(next_generation) < self.max_population_size:
                next_generation.append(child2)
        
        return np.array(next_generation[:self.max_population_size])

    def run(self):
        """
        Ejecuta el algoritmo genético.
        
        Retorna:
            tuple: (mejores coeficientes, mejor MSE)
        """
        population = self.initialize_population()
        for generation in range(self.generations):
            fitness, mse = self.compute_fitness(population)
            best_mse = np.min(mse)
            best_index = np.argmin(mse)
            best_individual = population[best_index]
            best_fitness = fitness[best_index]
            self.history_mse.append(best_mse)
            self.history_fitness.append(best_fitness)
            self.history_betas.append(best_individual.copy())
            
            # Selección
            selected = self.selection(population, fitness)
            # Crear próxima generación
            population = self.create_next_generation(selected)
            
            # Reportar progreso cada 50 generaciones
            if (generation + 1) % 50 == 0:
                print(f"Generación {generation+1}/{self.generations} - Mejor MSE: {best_mse:.4f} - Mejor Fitness: {best_fitness:.4f}")
        
        # Evaluar la mejor solución final
        fitness, mse = self.compute_fitness(population)
        best_mse = np.min(mse)
        best_index = np.argmin(mse)
        best_individual = population[best_index]
        return best_individual, best_mse
