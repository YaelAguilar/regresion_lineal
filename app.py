# app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
import threading

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import ttk

from ga import LinearRegressionGA

class Application:
    def __init__(self, master):
        self.master = master
        master.title("Regresión Lineal con Algoritmo Genético")
        master.geometry("1200x800")  # Ajusta el tamaño según tus necesidades

        self.dataset = None
        self.X = None
        self.Y = None
        self.num_features = 0
        self.ga = None
        self.Y_mean = 0
        self.Y_std = 1

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

        # Entrada para Probabilidad de Cruce
        Label(frame_params, text="Probabilidad de Cruce:").grid(row=0, column=0, sticky=E)
        self.entry_crossover_prob = Entry(frame_params)
        self.entry_crossover_prob.insert(0, "0.8")
        self.entry_crossover_prob.grid(row=0, column=1, padx=5, pady=2)

        # Entrada para Probabilidad de Mutación
        Label(frame_params, text="Probabilidad de Mutación:").grid(row=1, column=0, sticky=E)
        self.entry_mutation_prob = Entry(frame_params)
        self.entry_mutation_prob.insert(0, "0.1")
        self.entry_mutation_prob.grid(row=1, column=1, padx=5, pady=2)

        # Entrada para Probabilidad de Mutación del Gen
        Label(frame_params, text="Probabilidad de Mutación del Gen:").grid(row=2, column=0, sticky=E)
        self.entry_bit_mutation_prob = Entry(frame_params)
        self.entry_bit_mutation_prob.insert(0, "0.1")
        self.entry_bit_mutation_prob.grid(row=2, column=1, padx=5, pady=2)

        # Entrada para Población Inicial
        Label(frame_params, text="Población Inicial:").grid(row=3, column=0, sticky=E)
        self.entry_initial_population = Entry(frame_params)
        self.entry_initial_population.insert(0, "50")
        self.entry_initial_population.grid(row=3, column=1, padx=5, pady=2)

        # Entrada para Tamaño Máximo de Población
        Label(frame_params, text="Tamaño Máximo de Población:").grid(row=4, column=0, sticky=E)
        self.entry_max_population = Entry(frame_params)
        self.entry_max_population.insert(0, "100")
        self.entry_max_population.grid(row=4, column=1, padx=5, pady=2)

        # Entrada para Generaciones
        Label(frame_params, text="Generaciones:").grid(row=5, column=0, sticky=E)
        self.entry_generations = Entry(frame_params)
        self.entry_generations.insert(0, "500")
        self.entry_generations.grid(row=5, column=1, padx=5, pady=2)

        # Entrada para Tasa de Elitismo
        Label(frame_params, text="Tasa de Elitismo:").grid(row=6, column=0, sticky=E)
        self.entry_elitism_rate = Entry(frame_params)
        self.entry_elitism_rate.insert(0, "0.1")
        self.entry_elitism_rate.grid(row=6, column=1, padx=5, pady=2)

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

        # Predicción vs Real
        self.fig_pred_real = plt.Figure(figsize=(6,5), dpi=100, constrained_layout=True)
        self.ax_pred_real = self.fig_pred_real.add_subplot(111)
        self.canvas_pred_real = FigureCanvasTkAgg(self.fig_pred_real, master=self.tab_pred_real)
        self.canvas_pred_real.draw()

        frame_pred_real = Frame(self.tab_pred_real)
        frame_pred_real.pack(fill="both", expand=True)

        self.toolbar_pred_real = NavigationToolbar2Tk(self.canvas_pred_real, frame_pred_real)
        self.toolbar_pred_real.update()
        self.toolbar_pred_real.pack(side=TOP, fill="x")

        self.canvas_pred_real.get_tk_widget().pack(side=TOP, fill="both", expand=True)

        # Evolución del Fitness
        self.fig_fitness = plt.Figure(figsize=(6,5), dpi=100, constrained_layout=True)
        self.ax_fitness = self.fig_fitness.add_subplot(111)
        self.canvas_fitness = FigureCanvasTkAgg(self.fig_fitness, master=self.tab_fitness)
        self.canvas_fitness.draw()

        frame_fitness = Frame(self.tab_fitness)
        frame_fitness.pack(fill="both", expand=True)

        self.toolbar_fitness = NavigationToolbar2Tk(self.canvas_fitness, frame_fitness)
        self.toolbar_fitness.update()
        self.toolbar_fitness.pack(side=TOP, fill="x")

        self.canvas_fitness.get_tk_widget().pack(side=TOP, fill="both", expand=True)

        # Evolución de Betas
        self.fig_betas = plt.Figure(figsize=(6,5), dpi=100, constrained_layout=True)
        self.ax_betas = self.fig_betas.add_subplot(111)
        self.canvas_betas = FigureCanvasTkAgg(self.fig_betas, master=self.tab_betas)
        self.canvas_betas.draw()

        frame_betas = Frame(self.tab_betas)
        frame_betas.pack(fill="both", expand=True)

        self.toolbar_betas = NavigationToolbar2Tk(self.canvas_betas, frame_betas)
        self.toolbar_betas.update()
        self.toolbar_betas.pack(side=TOP, fill="x")

        self.canvas_betas.get_tk_widget().pack(side=TOP, fill="both", expand=True)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            try:
                self.dataset = pd.read_csv(file_path, sep=';')
                self.lbl_file.config(text=file_path.split('/')[-1])
                self.prepare_data()
                self.btn_run.config(state=NORMAL)
                messagebox.showinfo("Éxito", "Dataset cargado exitosamente.")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el archivo.\n{e}")

    def prepare_data(self):
        # Verificar estructura mínima
        if self.dataset.shape[1] < 3:
            raise ValueError("El dataset debe tener al menos una variable independiente (Beta) además de 'id' y 'Y'.")
        # Forzar la conversión de 'id' a numérico
        try:
            self.dataset['id'] = pd.to_numeric(self.dataset['id'], errors='raise')
        except ValueError:
            raise ValueError("La columna 'id' debe contener valores numéricos.")
        
        # Obtener los parámetros desde las entradas
        try:
            crossover_prob = float(self.entry_crossover_prob.get())
            if not (0 <= crossover_prob <=1):
                raise ValueError
        except ValueError:
            raise ValueError("La probabilidad de cruce debe ser un número entre 0 y 1.")
        
        try:
            mutation_prob = float(self.entry_mutation_prob.get())
            if not (0 <= mutation_prob <=1):
                raise ValueError
        except ValueError:
            raise ValueError("La probabilidad de mutación debe ser un número entre 0 y 1.")
        
        try:
            bit_mutation_prob = float(self.entry_bit_mutation_prob.get())
            if not (0 <= bit_mutation_prob <=1):
                raise ValueError
        except ValueError:
            raise ValueError("La probabilidad de mutación del gen debe ser un número entre 0 y 1.")
        
        try:
            initial_population_size = int(self.entry_initial_population.get())
            if initial_population_size <=0:
                raise ValueError
        except ValueError:
            raise ValueError("La población inicial debe ser un número positivo.")
        
        try:
            max_population_size = int(self.entry_max_population.get())
            if max_population_size <=0:
                raise ValueError
        except ValueError:
            raise ValueError("El tamaño máximo de población debe ser un número positivo.")
        
        # Validar que initial_population_size <= max_population_size
        if initial_population_size > max_population_size:
            raise ValueError("La población inicial no puede ser mayor que el tamaño máximo de población.")
        
        try:
            generations = int(self.entry_generations.get())
            if generations <=0:
                raise ValueError
        except ValueError:
            raise ValueError("El número de generaciones debe ser un número positivo.")
        
        try:
            elitism_rate = float(self.entry_elitism_rate.get())
            if not (0 <= elitism_rate <=1):
                raise ValueError
        except ValueError:
            raise ValueError("La tasa de elitismo debe ser un número entre 0 y 1.")
        
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.bit_mutation_prob = bit_mutation_prob
        self.initial_population_size = initial_population_size
        self.max_population_size = max_population_size
        self.elitism_rate = elitism_rate

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

        # Validar los parámetros antes de ejecutar
        try:
            # Los parámetros ya han sido validados en prepare_data
            pass
        except Exception as e:
            messagebox.showerror("Error", f"Error en los parámetros: {e}")
            return

        self.btn_run.config(state=DISABLED)
        self.txt_results.delete(1.0, END)
        self.txt_results.insert(END, "Ejecutando el ajuste de regresión...\n")

        # Ejecutar en un hilo separado para no congelar la GUI
        threading.Thread(target=self.execute_ga).start()

    def execute_ga(self):
        try:
            self.ga = LinearRegressionGA(
                X=self.X,
                Y=self.Y,
                generations=self.generations,
                crossover_prob=self.crossover_prob,
                mutation_prob=self.mutation_prob,
                bit_mutation_prob=self.bit_mutation_prob,
                initial_population_size=self.initial_population_size,
                max_population_size=self.max_population_size,
                elitism_rate=self.elitism_rate
            )
            best_weights, best_mse = self.ga.run()
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
        self.fig_pred_real.tight_layout()
        self.canvas_pred_real.draw()

        # Graficar Evolución del Fitness
        self.ax_fitness.clear()
        self.ax_fitness.plot(range(1, len(self.ga.history_fitness) +1), self.ga.history_fitness, label="Mejor Fitness", color='purple')
        self.ax_fitness.set_xlabel("Generaciones")
        self.ax_fitness.set_ylabel("Mejor Fitness")
        self.ax_fitness.set_title("Evolución del Fitness")
        self.ax_fitness.legend()
        self.ax_fitness.grid(True)
        self.fig_fitness.tight_layout()
        self.canvas_fitness.draw()

        # Graficar Evolución de Betas
        self.ax_betas.clear()
        history_betas = np.array(self.ga.history_betas)
        for i in range(history_betas.shape[1]):
            self.ax_betas.plot(range(1, len(history_betas)+1), history_betas[:,i], label=self.feature_names[i])
        self.ax_betas.set_xlabel("Generaciones")
        self.ax_betas.set_ylabel("Valores de Beta")
        self.ax_betas.set_title("Evolución de Betas")
        self.ax_betas.legend()
        self.ax_betas.grid(True)
        self.fig_betas.tight_layout()
        self.canvas_betas.draw()

        self.btn_run.config(state=NORMAL)
        messagebox.showinfo("Éxito", "Ajuste de regresión completado.")
