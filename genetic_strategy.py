from abc import ABC, abstractmethod

class GeneticStrategy(ABC):
    @abstractmethod
    def setup_parameters(self, x_min, x_max, dx, min_population, max_population):
        pass

    @abstractmethod
    def initialize_population(self):
        pass

    @abstractmethod
    def fitness(self, individual):
        pass

    @abstractmethod
    def get_population_stats(self, population):
        pass

    @abstractmethod
    def select_best(self, population):
        pass

    @abstractmethod
    def crossover(self, population):
        pass

    @abstractmethod
    def mutate(self, population):
        pass

    @abstractmethod
    def get_best_and_worst(self, population):
        pass

    @abstractmethod
    def decode_solution(self, binary_string):
        pass

    @abstractmethod
    def pruning(self, population, offspring):
        pass
