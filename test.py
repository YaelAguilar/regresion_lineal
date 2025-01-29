import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
import threading
import random

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import ttk

class LinearRegressionGA:
    def __init__(self, X, Y, learning_rate=0.01, generations=500, crossover_prob=0.8, mutation_prob=0.1, elitism_rate=0.1):
        self.X = X  # Matriz de características con intercepto
        self.Y = Y  # Vector de variables dependientes normalizado
        self.learning_rate = learning_rate
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism_rate = elitism_rate
        self.num_features = X.shape[1]
        self.population_size = 100  # Tamaño de población aumentado
        self.history_mse = []
        self.history_fitness = []  # Almacena el mejor fitness de cada generación
        self.history_betas = []

    def initialize_population(self):
        # Inicializa la población con valores aleatorios entre -10 y 10
        return np.random.uniform(-10, 10, (self.population_size, self.num_features))

    def compute_fitness(self, population):
        # Fitness inverso al MSE
        predictions = population @ self.X.T
        mse = np.mean((predictions - self.Y) ** 2, axis=1)
        fitness = 1 / (mse + 1e-10)  # Añadir epsilon para evitar división por cero
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
            # Imprimir progreso cada 50 generaciones
            if (generation + 1) % 50 == 0:
                print(f"Generación {generation+1}/{self.generations} - Mejor MSE: {best_mse:.4f} - Mejor Fitness: {best_fitness:.4f}")
        # Al final, obtener el mejor individuo
        fitness, mse = self.compute_fitness(population)
        best_mse = np.min(mse)
        best_index = np.argmin(mse)
        best_individual = population[best_index]
        return best_individual, best_mse

class Application:
    def __init__(self, master):
        self.master = master
        master.title("Regresión Lineal con Algoritmo Genético")

        # Variables
        self.dataset = None
        self.X = None
        self.Y = None
        self.num_features = 0
        self.ga = None
        self.Y_mean = 0
        self.Y_std = 1

        # Crear componentes de la GUI
        self.create_widgets()

    def create_widgets(self):
        # Frame para carga de datos
        frame_data = LabelFrame(self.master, text="Cargar Dataset", padx=10, pady=10)
        frame_data.pack(padx=10, pady=5, fill="x")

        self.btn_load = Button(frame_data, text="Cargar CSV", command=self.load_csv)
        self.btn_load.pack(side=LEFT)

        self.lbl_file = Label(frame_data, text="Ningún archivo cargado")
        self.lbl_file.pack(side=LEFT, padx=10)

        # Frame para parámetros
        frame_params = LabelFrame(self.master, text="Parámetros", padx=10, pady=10)
        frame_params.pack(padx=10, pady=5, fill="x")

        # Entrada para Tasa de Aprendizaje
        Label(frame_params, text="Tasa de Aprendizaje:").grid(row=0, column=0, sticky=E)
        self.entry_learning_rate = Entry(frame_params)
        self.entry_learning_rate.insert(0, "0.01")  # Valor predeterminado reducido
        self.entry_learning_rate.grid(row=0, column=1, padx=5, pady=2)

        # Entrada para Generaciones
        Label(frame_params, text="Generaciones:").grid(row=1, column=0, sticky=E)
        self.entry_generations = Entry(frame_params)
        self.entry_generations.insert(0, "500")  # Valor predeterminado
        self.entry_generations.grid(row=1, column=1, padx=5, pady=2)

        # Frame para ejecutar
        frame_run = Frame(self.master, padx=10, pady=10)
        frame_run.pack(padx=10, pady=5, fill="x")

        self.btn_run = Button(frame_run, text="Ejecutar Ajuste", command=self.run_regression, state=DISABLED)
        self.btn_run.pack()

        # Frame para resultados
        frame_results = LabelFrame(self.master, text="Resultados", padx=10, pady=10)
        frame_results.pack(padx=10, pady=5, fill="both", expand=True)

        self.txt_results = Text(frame_results, height=10)
        self.txt_results.pack(fill="both", expand=True)

        # Frame para gráficas
        frame_plots = LabelFrame(self.master, text="Gráficas", padx=10, pady=10)
        frame_plots.pack(padx=10, pady=5, fill="both", expand=True)

        # Utilizar pestañas para las gráficas
        self.tabs = ttk.Notebook(frame_plots)
        self.tabs.pack(fill="both", expand=True)

        # Pestañas
        self.tab_pred_real = ttk.Frame(self.tabs)
        self.tab_fitness = ttk.Frame(self.tabs)
        self.tab_betas = ttk.Frame(self.tabs)

        self.tabs.add(self.tab_pred_real, text="Predicción vs Real")
        self.tabs.add(self.tab_fitness, text="Evolución del Fitness")
        self.tabs.add(self.tab_betas, text="Evolución de Betas")

        # Figuras y Barras de Herramientas

        # Predicción vs Real
        self.fig_pred_real = plt.Figure(figsize=(6,5), dpi=100)
        self.ax_pred_real = self.fig_pred_real.add_subplot(111)
        self.canvas_pred_real = FigureCanvasTkAgg(self.fig_pred_real, master=self.tab_pred_real)
        self.canvas_pred_real.draw()

        # Frame para Predicción vs Real (Canvas + Toolbar)
        frame_pred_real = Frame(self.tab_pred_real)
        frame_pred_real.pack(fill="both", expand=True)

        # Barra de herramientas para Predicción vs Real
        self.toolbar_pred_real = NavigationToolbar2Tk(self.canvas_pred_real, frame_pred_real)
        self.toolbar_pred_real.update()
        self.toolbar_pred_real.pack(side=TOP, fill="x")

        # Canvas para Predicción vs Real
        self.canvas_pred_real.get_tk_widget().pack(side=TOP, fill="both", expand=True)

        # Evolución del Fitness
        self.fig_fitness = plt.Figure(figsize=(6,5), dpi=100)
        self.ax_fitness = self.fig_fitness.add_subplot(111)
        self.canvas_fitness = FigureCanvasTkAgg(self.fig_fitness, master=self.tab_fitness)
        self.canvas_fitness.draw()

        # Frame para Evolución del Fitness (Canvas + Toolbar)
        frame_fitness = Frame(self.tab_fitness)
        frame_fitness.pack(fill="both", expand=True)

        # Barra de herramientas para Evolución del Fitness
        self.toolbar_fitness = NavigationToolbar2Tk(self.canvas_fitness, frame_fitness)
        self.toolbar_fitness.update()
        self.toolbar_fitness.pack(side=TOP, fill="x")

        # Canvas para Evolución del Fitness
        self.canvas_fitness.get_tk_widget().pack(side=TOP, fill="both", expand=True)

        # Evolución de Betas
        self.fig_betas = plt.Figure(figsize=(6,5), dpi=100)
        self.ax_betas = self.fig_betas.add_subplot(111)
        self.canvas_betas = FigureCanvasTkAgg(self.fig_betas, master=self.tab_betas)
        self.canvas_betas.draw()

        # Frame para Evolución de Betas (Canvas + Toolbar)
        frame_betas = Frame(self.tab_betas)
        frame_betas.pack(fill="both", expand=True)

        # Barra de herramientas para Evolución de Betas
        self.toolbar_betas = NavigationToolbar2Tk(self.canvas_betas, frame_betas)
        self.toolbar_betas.update()
        self.toolbar_betas.pack(side=TOP, fill="x")

        # Canvas para Evolución de Betas
        self.canvas_betas.get_tk_widget().pack(side=TOP, fill="both", expand=True)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            try:
                self.dataset = pd.read_csv(file_path, sep=';')
                self.lbl_file.config(text=file_path.split('/')[-1])
                # Preparar X e Y
                self.prepare_data()
                self.btn_run.config(state=NORMAL)
                messagebox.showinfo("Éxito", "Dataset cargado exitosamente.")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el archivo.\n{e}")

    def prepare_data(self):
        # Verificar estructura mínima
        if self.dataset.shape[1] < 3:
            raise ValueError("El dataset debe tener al menos una variable independiente (Beta) además de 'id' y 'Y'.")
        # Asumiendo que la primera columna es 'id' y la última es 'Y'
        try:
            # Forzar la conversión de 'id' a numérico
            self.dataset['id'] = pd.to_numeric(self.dataset['id'], errors='raise')
        except ValueError:
            raise ValueError("La columna 'id' debe contener valores numéricos.")
        
        # Obtener el número de generaciones de la entrada del usuario
        self.generations = int(self.entry_generations.get())
        if self.generations <= 0:
            raise ValueError("El número de generaciones debe ser un número positivo.")
        
        self.num_features = self.dataset.shape[1] - 2  # Excluyendo 'id' y 'Y'
        X = self.dataset.iloc[:, 1:-1].values  # Todas las columnas de X
        Y = self.dataset.iloc[:, -1].values    # Última columna 'Y'
        
        # Normalizar X para mejorar el desempeño del GA
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Evitar división por cero
        X_normalized = (X - X_mean) / X_std
        
        # Normalizar Y
        self.Y_mean = np.mean(Y)
        self.Y_std = np.std(Y)
        if self.Y_std == 0:
            self.Y_std = 1  # Evitar división por cero
        Y_normalized = (Y - self.Y_mean) / self.Y_std
        
        # Añadir columna de 1 para el intercepto
        self.X = np.hstack((np.ones((X_normalized.shape[0],1)), X_normalized))
        self.Y = Y_normalized
        self.feature_names = ['Intercept'] + list(self.dataset.columns[1:-1])
        self.X_mean = X_mean
        self.X_std = X_std

    def run_regression(self):
        if self.X is None or self.Y is None:
            messagebox.showerror("Error", "No se ha cargado ningún dataset.")
            return

        try:
            learning_rate = float(self.entry_learning_rate.get())
            if learning_rate <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "La tasa de aprendizaje debe ser un número positivo.")
            return

        try:
            generations = int(self.entry_generations.get())
            if generations <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "El número de generaciones debe ser un número positivo.")
            return

        # Deshabilitar el botón mientras se ejecuta
        self.btn_run.config(state=DISABLED)
        self.txt_results.delete(1.0, END)
        self.txt_results.insert(END, "Ejecutando el ajuste de regresión...\n")

        # Ejecutar en un hilo separado para no congelar la GUI
        threading.Thread(target=self.execute_ga, args=(learning_rate, generations)).start()

    def execute_ga(self, learning_rate, generations):
        try:
            self.ga = LinearRegressionGA(
                X=self.X,
                Y=self.Y,
                learning_rate=learning_rate,
                generations=generations,
                crossover_prob=0.8,
                mutation_prob=0.1,
                elitism_rate=0.1
            )
            best_weights, best_mse = self.ga.run()
            # Actualizar resultados en la GUI
            self.master.after(0, self.display_results, best_weights, best_mse)
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error durante la ejecución.\n{e}")
            self.btn_run.config(state=NORMAL)

    def display_results(self, best_weights, best_mse):
        # Desnormalizar los pesos
        intercept = (best_weights[0] * self.Y_std) - (self.Y_std * np.sum((best_weights[1:] * self.X_mean) / self.X_std)) + self.Y_mean
        betas = (best_weights[1:] * self.Y_std) / self.X_std
        adjusted_weights = np.concatenate(([intercept], betas))

        # Mostrar MSE y Betas
        self.txt_results.insert(END, f"Error Cuadrático Medio (MSE): {best_mse * (self.Y_std ** 2):.2f}\n")
        self.txt_results.insert(END, "Pesos (Betas):\n")
        for name, weight in zip(self.feature_names, adjusted_weights):
            self.txt_results.insert(END, f"  {name}: {weight:.6f}\n")

        # Obtener índices para el eje X
        indices = self.dataset['id'].values  # Asumiendo que 'id' es secuencial desde 1 hasta N

        # Graficar Predicción vs Real
        predictions_normalized = self.X @ best_weights
        predictions = predictions_normalized * self.Y_std + self.Y_mean
        Y_real = self.dataset.iloc[:, -1].values  # Valores reales originales

        self.ax_pred_real.clear()
        self.ax_pred_real.plot(indices, Y_real, label="Real Y", color='green', marker='o')
        self.ax_pred_real.plot(indices, predictions, label="Predicción Y", color='blue', marker='x')
        self.ax_pred_real.set_xlabel("Índice")
        self.ax_pred_real.set_ylabel("Valor de Y")
        self.ax_pred_real.set_title("Predicción vs Real")
        self.ax_pred_real.legend()
        self.ax_pred_real.grid(True)
        self.canvas_pred_real.draw()

        # Graficar Evolución del Fitness
        self.ax_fitness.clear()
        self.ax_fitness.plot(range(1, self.ga.generations +1), self.ga.history_fitness, label="Mejor Fitness", color='purple')
        self.ax_fitness.set_xlabel("Generaciones")
        self.ax_fitness.set_ylabel("Mejor Fitness")
        self.ax_fitness.set_title("Evolución del Fitness")
        self.ax_fitness.legend()
        self.ax_fitness.grid(True)
        self.canvas_fitness.draw()

        # Graficar Evolución de Betas
        self.ax_betas.clear()
        history_betas = np.array(self.ga.history_betas)
        for i in range(history_betas.shape[1]):
            self.ax_betas.plot(range(1, self.ga.generations +1), history_betas[:,i], label=self.feature_names[i])
        self.ax_betas.set_xlabel("Generaciones")
        self.ax_betas.set_ylabel("Valores de Beta")
        self.ax_betas.set_title("Evolución de Betas")
        self.ax_betas.legend()
        self.ax_betas.grid(True)
        self.canvas_betas.draw()

        self.btn_run.config(state=NORMAL)
        messagebox.showinfo("Éxito", "Ajuste de regresión completado.")

def main():
    root = Tk()
    app = Application(root)
    root.mainloop()

if __name__ == "__main__":
    main()
