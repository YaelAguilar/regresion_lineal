import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import cv2
import tempfile
import shutil
import csv

from ga import BasicGeneticStrategy
from objective import mean_squared_error

# Para que matplotlib funcione dentro de tkinter
matplotlib.use("TkAgg")

class GeneticAlgorithmApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Algoritmo Genético - Regresión Lineal")
        self.root.geometry("1600x1000")

        # Configuración de filas y columnas para el layout de Tkinter
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)

        # Variables para almacenar datos de generaciones y animación
        self.generations_pop = []
        self.generations_fit = []
        self.generations_betas = []
        self.anim = None

        self.is_paused = False
        self.current_frame = 0

        # Directorio temporal para guardar frames de la animación
        self.temp_dir = tempfile.mkdtemp()
        print(f"Directorio temporal creado: {self.temp_dir}")

        # Variables para almacenar los datos cargados
        self.X = np.array([])
        self.Y = np.array([])

        # Configuración de los parámetros del Algoritmo Genético en la GUI
        frame_params = ttk.LabelFrame(self.root, text="Parámetros del AG", padding=10)
        frame_params.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Entradas para parámetros
        ttk.Label(frame_params, text="Rango mínimo de betas:").grid(row=0, column=0, sticky="e")
        self.entry_beta_min = ttk.Entry(frame_params, width=10)
        self.entry_beta_min.grid(row=0, column=1, padx=5, pady=5)
        self.entry_beta_min.insert(0, "-1000")

        ttk.Label(frame_params, text="Rango máximo de betas:").grid(row=1, column=0, sticky="e")
        self.entry_beta_max = ttk.Entry(frame_params, width=10)
        self.entry_beta_max.grid(row=1, column=1, padx=5, pady=5)
        self.entry_beta_max.insert(0, "1000")

        ttk.Label(frame_params, text="Número de generaciones:").grid(row=2, column=0, sticky="e")
        self.entry_num_gen = ttk.Entry(frame_params, width=10)
        self.entry_num_gen.grid(row=2, column=1, padx=5, pady=5)
        self.entry_num_gen.insert(0, "50")

        ttk.Label(frame_params, text="Tamaño de la población inicial:").grid(row=3, column=0, sticky="e")
        self.entry_pop_init = ttk.Entry(frame_params, width=10)
        self.entry_pop_init.grid(row=3, column=1, padx=5, pady=5)
        self.entry_pop_init.insert(0, "20")

        ttk.Label(frame_params, text="Tamaño de la población máxima:").grid(row=4, column=0, sticky="e")
        self.entry_pop_max = ttk.Entry(frame_params, width=10)
        self.entry_pop_max.grid(row=4, column=1, padx=5, pady=5)
        self.entry_pop_max.insert(0, "50")

        ttk.Label(frame_params, text="Probabilidad de cruza:").grid(row=5, column=0, sticky="e")
        self.entry_p_cruza = ttk.Entry(frame_params, width=10)
        self.entry_p_cruza.grid(row=5, column=1, padx=5, pady=5)
        self.entry_p_cruza.insert(0, "0.8")

        ttk.Label(frame_params, text="Probabilidad de mutación de individuo:").grid(row=6, column=0, sticky="e")
        self.entry_p_mut_ind = ttk.Entry(frame_params, width=10)
        self.entry_p_mut_ind.grid(row=6, column=1, padx=5, pady=5)
        self.entry_p_mut_ind.insert(0, "0.1")

        ttk.Label(frame_params, text="Probabilidad de mutación de bit:").grid(row=7, column=0, sticky="e")
        self.entry_p_mut_bit = ttk.Entry(frame_params, width=10)
        self.entry_p_mut_bit.grid(row=7, column=1, padx=5, pady=5)
        self.entry_p_mut_bit.insert(0, "0.01")

        # Botón para cargar archivo de datos
        self.load_button = ttk.Button(frame_params, text="Cargar Archivo de Datos", command=self.load_file)
        self.load_button.grid(row=8, column=0, columnspan=2, pady=10)

        # Botón para ejecutar el Algoritmo Genético
        self.run_button = ttk.Button(frame_params, text="Ejecutar AG", command=self.run_ga, state="disabled")
        self.run_button.grid(row=9, column=0, columnspan=2, pady=10)

        # Frame para mostrar parámetros de codificación
        frame_codif = ttk.LabelFrame(self.root, text="Parámetros de codificación", padding=10)
        frame_codif.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.label_num_betas = ttk.Label(frame_codif, text="Número de betas: 4")
        self.label_num_betas.pack(anchor="w", pady=2)

        self.label_bits_per_beta = ttk.Label(frame_codif, text="Número de bits por beta: 16")
        self.label_bits_per_beta.pack(anchor="w", pady=2)

        self.label_total_bits = ttk.Label(frame_codif, text="Total de bits por individuo: 64")
        self.label_total_bits.pack(anchor="w", pady=2)

        # Frame para mostrar soluciones y evolución
        frame_solutions = ttk.LabelFrame(self.root, text="Solución y evolución", padding=10)
        frame_solutions.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        # Frame para mostrar la mejor solución encontrada
        frame_best_solution = ttk.LabelFrame(frame_solutions, text="Mejor Solución Encontrada", padding=10)
        frame_best_solution.pack(fill="x", expand=False, padx=5, pady=5)

        # Labels para mostrar detalles de la mejor solución
        self.label_best_betas = ttk.Label(frame_best_solution, text="Betas: ")
        self.label_best_betas.pack(anchor="w", pady=2)

        self.label_best_mse = ttk.Label(frame_best_solution, text="Mean Squared Error: ")
        self.label_best_mse.pack(anchor="w", pady=2)

        # Frame para mostrar la tabla de evolución de generaciones
        frame_table = ttk.LabelFrame(frame_solutions, text="Evaluación de Generaciones", padding=10)
        frame_table.pack(fill="both", expand=True, padx=5, pady=5)

        # Configuración de la tabla con columnas específicas
        columns = ("Generación", "MSE", "Beta0", "Beta1", "Beta2", "Beta3")
        self.tree = ttk.Treeview(frame_table, columns=columns, show="headings", height=8)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)

        # Scrollbar para la tabla
        scrollbar = ttk.Scrollbar(frame_table, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # Frame para contener las gráficas
        self.root.grid_rowconfigure(1, weight=1)
        frame_plots_container = ttk.Frame(self.root)
        frame_plots_container.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

        self.canvas_plots = tk.Canvas(frame_plots_container)
        self.canvas_plots.pack(side="left", fill="both", expand=True)

        self.scrollbar_plots = ttk.Scrollbar(frame_plots_container, orient="vertical",
                                             command=self.canvas_plots.yview)
        self.scrollbar_plots.pack(side="right", fill="y")

        self.canvas_plots.configure(yscrollcommand=self.scrollbar_plots.set)

        # Frame interno para las gráficas
        self.frame_plots = ttk.LabelFrame(self.canvas_plots, text="Gráficas", padding=10)
        self.canvas_plots_window = self.canvas_plots.create_window((0, 0),
                                                                   window=self.frame_plots,
                                                                   anchor="nw")

        # Configuración de eventos para ajustar el scroll
        self.frame_plots.bind("<Configure>", self.on_frame_plots_configure)
        self.canvas_plots.bind("<Configure>", self.on_canvas_plots_configure)

        # Figura 1: Evolución de las betas
        self.fig1 = plt.Figure(figsize=(8, 4), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.frame_plots)
        self.canvas1.get_tk_widget().pack(side="top", fill="both", expand=True, padx=5, pady=5)

        # Figura 2: Evolución del fitness (MSE)
        self.fig2 = plt.Figure(figsize=(8, 4), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.frame_plots)
        self.canvas2.get_tk_widget().pack(side="top", fill="both", expand=True, padx=5, pady=5)

        # Figura 3: Evolución de los betas y fitness (Animación)
        self.fig3 = plt.Figure(figsize=(8, 4), dpi=100)
        self.ax3 = self.fig3.add_subplot(111)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.frame_plots)
        self.canvas3.get_tk_widget().pack(side="top", fill="both", expand=True, padx=5, pady=5)

        # Controles de la animación (Retroceder, Avanzar, Replay)
        frame_animation_controls = ttk.Frame(self.frame_plots)
        frame_animation_controls.pack(side="top", pady=10)

        # Botón Retroceder
        self.prev_icon = "⏮️"
        self.prev_button = ttk.Button(frame_animation_controls, text=self.prev_icon, command=self.prev_frame)
        self.prev_button.grid(row=0, column=0, padx=5)

        # Botón Avanzar
        self.next_icon = "⏭️"
        self.next_button = ttk.Button(frame_animation_controls, text=self.next_icon, command=self.next_frame)
        self.next_button.grid(row=0, column=1, padx=5)

        # Botón Replay
        self.replay_icon = "🔁"
        self.replay_button = ttk.Button(frame_animation_controls, text=self.replay_icon, command=self.replay_animation)
        self.replay_button.grid(row=0, column=2, padx=5)

        # Label para mostrar la generación actual
        self.generation_label = ttk.Label(frame_animation_controls, text="Generación: 0")
        self.generation_label.grid(row=0, column=3, padx=10)

        # Deshabilitar controles de animación al inicio
        self.disable_animation_controls()

    def load_file(self):
        """
        Open a file dialog to select and load the dataset.
        """
        file_path = filedialog.askopenfilename(
            title="Seleccionar Archivo de Datos",
            filetypes=(("Archivos CSV", "*.csv"), ("Todos los Archivos", "*.*"))
        )
        if not file_path:
            return  # El usuario canceló la selección

        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                header = next(reader)  # Leer encabezado
                if len(header) < 5:
                    raise ValueError("El archivo no tiene suficientes columnas.")

                # Identificar índices de columnas
                id_idx = header.index("id")
                x1_idx = header.index("X1")
                x2_idx = header.index("X2")
                x3_idx = header.index("X3")
                y_idx = header.index("Y")

                X = []
                Y = []
                for row in reader:
                    if len(row) < 5:
                        continue  # Saltar filas mal formateadas
                    try:
                        x1 = float(row[x1_idx])
                        x2 = float(row[x2_idx])
                        x3 = float(row[x3_idx])
                        y = float(row[y_idx])
                        X.append([x1, x2, x3])
                        Y.append(y)
                    except ValueError:
                        continue  # Saltar filas con datos inválidos

            if not X or not Y:
                raise ValueError("No se encontraron datos válidos en el archivo.")

            self.X = np.array(X)
            self.Y = np.array(Y)
            print(f"Dataset cargado con éxito: {len(self.Y)} instancias.")

            messagebox.showinfo("Éxito", f"Archivo cargado correctamente:\n{file_path}")

            # Habilitar el botón de ejecutar AG
            self.run_button.config(state="normal")

        except Exception as e:
            messagebox.showerror("Error al Cargar Archivo", f"No se pudo cargar el archivo:\n{e}")

    def on_frame_plots_configure(self, event):
        self.canvas_plots.configure(scrollregion=self.canvas_plots.bbox("all"))

    def on_canvas_plots_configure(self, event):
        canvas_width = event.width
        self.canvas_plots.itemconfig(self.canvas_plots_window, width=canvas_width)

    def run_ga(self):
        # Leer parámetros desde la GUI
        try:
            beta_min = float(self.entry_beta_min.get())
            beta_max = float(self.entry_beta_max.get())
            num_gen = int(self.entry_num_gen.get())
            pop_init = int(self.entry_pop_init.get())
            pop_max = int(self.entry_pop_max.get())
            p_cruza = float(self.entry_p_cruza.get())
            p_mut_ind = float(self.entry_p_mut_ind.get())
            p_mut_bit = float(self.entry_p_mut_bit.get())
        except ValueError:
            messagebox.showerror("Error de Entrada", "Por favor, ingresa valores válidos.")
            return

        # Validaciones de los parámetros ingresados
        if beta_min >= beta_max:
            messagebox.showerror("Error de Rango", "El rango mínimo de betas debe ser menor que el rango máximo.")
            return

        if pop_init <= 0 or pop_max <= 0 or num_gen <=0:
            messagebox.showerror("Error de Parámetros", "Los tamaños de población y el número de generaciones deben ser positivos.")
            return

        if pop_init > pop_max:
            messagebox.showerror("Error de Parámetros", "El tamaño de población inicial no puede ser mayor que el tamaño máximo.")
            return

        if self.X.size ==0 or self.Y.size ==0:
            messagebox.showerror("Error de Datos", "El dataset no está cargado o está vacío.")
            return

        # Inicializar la estrategia genética
        genetic_strategy = BasicGeneticStrategy(
            function=mean_squared_error,
            X=self.X,
            Y=self.Y,
            num_betas=4,
            bits_per_beta=16,
            crossover_prob=p_cruza,
            mutation_prob=p_mut_ind,
            bit_mutation_prob=p_mut_bit,
            elitism_rate=0.1
        )
        genetic_strategy.setup_parameters(beta_min, beta_max, dx=0.1, min_population=pop_init, max_population=pop_max)

        # Calcular parámetros de codificación
        num_betas = genetic_strategy.num_betas
        bits_per_beta = genetic_strategy.bits_per_beta
        total_bits = genetic_strategy.total_bits

        self.num_betas = num_betas
        self.bits_per_beta = bits_per_beta

        # Actualizar etiquetas de parámetros de codificación en la GUI
        self.label_num_betas.config(text=f"Número de betas: {num_betas}")
        self.label_bits_per_beta.config(text=f"Número de bits por beta: {bits_per_beta}")
        self.label_total_bits.config(text=f"Total de bits por individuo: {total_bits}")

        # Generar población inicial
        population = genetic_strategy.initialize_population()

        # Limpiar historial de generaciones anteriores
        self.generations_pop = []
        self.generations_fit = []
        self.generations_betas = []

        # Datos para la tabla de evolución
        table_data = []

        # Evaluar población inicial y añadir como generación 0
        fitness_values = [genetic_strategy.fitness(ind) for ind in population]
        betas_values = [genetic_strategy.decode_solution(ind) for ind in population]
        self.generations_pop.append(population.copy())
        
        # Determinar el mejor individuo de la generación 0
        best_idx = np.argmax(fitness_values)
        best_ind = population[best_idx]
        best_betas = betas_values[best_idx]
        best_mse = mean_squared_error(best_betas, genetic_strategy.X, genetic_strategy.Y)

        # Almacenar solo las betas y fitness del mejor individuo
        self.generations_fit.append(fitness_values[best_idx])
        self.generations_betas.append(best_betas.copy())

        table_data.append([0, f"{best_mse:.6f}", 
                           f"{best_betas[0]:.4f}", 
                           f"{best_betas[1]:.4f}", 
                           f"{best_betas[2]:.4f}", 
                           f"{best_betas[3]:.4f}"])

        # Ejecutar el algoritmo genético para cada generación
        for gen in range(1, num_gen):
            # Reproducción
            selected = genetic_strategy.select_best(population)
            offspring_crossover, num_pairs = genetic_strategy.crossover(selected)
            offspring_mutated, total_mutations, total_mutated_bits = genetic_strategy.mutate(offspring_crossover)
            population = genetic_strategy.pruning(selected, offspring_mutated)

            # Evaluar nueva población
            fitness_values = [genetic_strategy.fitness(ind) for ind in population]
            betas_values = [genetic_strategy.decode_solution(ind) for ind in population]
            self.generations_pop.append(population.copy())

            # Determinar el mejor individuo de la generación actual
            best_idx = np.argmax(fitness_values)
            best_ind = population[best_idx]
            best_betas = betas_values[best_idx]
            best_mse = mean_squared_error(best_betas, genetic_strategy.X, genetic_strategy.Y)

            # Almacenar solo las betas y fitness del mejor individuo
            self.generations_fit.append(fitness_values[best_idx])
            self.generations_betas.append(best_betas.copy())

            table_data.append([gen, f"{best_mse:.6f}", 
                               f"{best_betas[0]:.4f}", 
                               f"{best_betas[1]:.4f}", 
                               f"{best_betas[2]:.4f}", 
                               f"{best_betas[3]:.4f}"])

        # Verificar si hay al menos una generación válida
        if not self.generations_betas:
            messagebox.showerror("Error de Datos", "No hay generaciones válidas para graficar.")
            return

        # Llenar la tabla de evolución en la GUI
        for row in self.tree.get_children():
            self.tree.delete(row)

        for row_data in table_data:
            g, mse, b0, b1, b2, b3 = row_data
            self.tree.insert("", tk.END, values=(
                g, mse, b0, b1, b2, b3
            ))

        # Mostrar la mejor solución final en la GUI
        best_index_final = np.argmax(fitness_values)
        best_ind_final = population[best_index_final]
        best_betas_final = betas_values[best_index_final]
        best_mse_final = mean_squared_error(best_betas_final, genetic_strategy.X, genetic_strategy.Y)

        betas_str = ", ".join([f"beta{i}: {beta:.4f}" for i, beta in enumerate(best_betas_final)])
        self.label_best_betas.config(text=f"Betas: {betas_str}")
        self.label_best_mse.config(text=f"Mean Squared Error: {best_mse_final:.6f}")

        self.genetic_strategy = genetic_strategy

        # Generar y mostrar las gráficas
        self.plot_betas_evolution()
        self.plot_fitness_evolution()
        self.plot_animation()

        # Habilitar controles de animación
        self.enable_animation_controls()

    def plot_betas_evolution(self):
        self.ax1.clear()

        # Evolución de las betas
        generations = range(len(self.generations_betas))
        betas_over_gen = np.array(self.generations_betas)  # shape: (num_generations, num_betas)

        for beta_idx in range(self.num_betas):
            self.ax1.plot(generations, betas_over_gen[:, beta_idx], label=f"Beta{beta_idx}")

        self.ax1.set_title("Evolución de las Betas")
        self.ax1.set_xlabel("Generaciones")
        self.ax1.set_ylabel("Valor de Beta")
        self.ax1.legend()
        self.canvas1.draw()

    def plot_fitness_evolution(self):
        self.ax2.clear()

        # Evolución del fitness (MSE)
        generations = range(len(self.generations_fit))
        best_fitness = self.generations_fit  # Ahora solo almacena el fitness del mejor individuo por generación
        avg_fitness = [np.mean(self.generations_fit)] * len(self.generations_fit)  # Promedio constante
        worst_fitness = [min(self.generations_fit)] * len(self.generations_fit)  # Peor fitness constante

        # Convert fitness back to MSE for plotting
        best_mse = [1 / fit -1 for fit in best_fitness]
        avg_mse = [1 / fit -1 for fit in avg_fitness]
        worst_mse = [1 / fit -1 for fit in worst_fitness]

        self.ax2.plot(generations, best_mse, label="Mejor MSE", color="green")
        self.ax2.plot(generations, avg_mse, label="Promedio MSE", color="blue")
        self.ax2.plot(generations, worst_mse, label="Peor MSE", color="red")

        self.ax2.set_title("Evolución del Fitness (MSE)")
        self.ax2.set_xlabel("Generaciones")
        self.ax2.set_ylabel("Mean Squared Error")
        self.ax2.legend()
        self.canvas2.draw()

    def plot_animation(self):
        self.ax3.clear()

        # Configurar el gráfico
        self.ax3.set_title("Evolución de las Betas y Fitness (Animación)")
        self.ax3.set_xlabel("Generaciones")
        self.ax3.set_ylabel("Valores")

        # Inicializar líneas para cada beta y para el fitness (MSE)
        self.lines_betas = []
        colors = ["blue", "green", "red", "purple"]
        for beta_idx in range(self.num_betas):
            line, = self.ax3.plot([], [], label=f"Beta{beta_idx}", color=colors[beta_idx])
            self.lines_betas.append(line)
        self.line_fitness, = self.ax3.plot([], [], label="Mejor MSE", color="orange")

        self.ax3.legend()

        # Detener cualquier animación previa
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass

        self.is_paused = False
        self.current_frame = 0
        self.generation_label.config(text="Generación: 0")

        # Limpiar el directorio temporal antes de crear nuevos frames
        shutil.rmtree(self.temp_dir)
        self.temp_dir = tempfile.mkdtemp()
        print(f"Directorio temporal limpiado y recreado: {self.temp_dir}")

        def init():
            for line in self.lines_betas:
                line.set_data([], [])
            self.line_fitness.set_data([], [])
            self.generation_label.config(text="Generación: 0")
            return self.lines_betas + [self.line_fitness]

        def update(frame):
            if self.is_paused:
                return self.lines_betas + [self.line_fitness]

            if frame >= len(self.generations_betas):
                if self.anim is not None:
                    self.anim.event_source.stop()
                return self.lines_betas + [self.line_fitness]

            # Actualizar líneas de betas
            generations = range(frame +1)
            betas_over_gen = np.array(self.generations_betas[:frame +1])
            fitness_over_gen = [1 / fit -1 for fit in self.generations_fit[:frame +1]]

            for beta_idx in range(self.num_betas):
                self.lines_betas[beta_idx].set_data(generations, betas_over_gen[:, beta_idx])

            # Actualizar línea de fitness
            self.line_fitness.set_data(generations, fitness_over_gen)

            # Ajustar límites del gráfico
            self.ax3.set_xlim(0, len(self.generations_betas))
            all_betas = betas_over_gen.flatten()
            min_beta = min(all_betas)
            max_beta = max(all_betas)
            min_fitness = min(fitness_over_gen)
            max_fitness = max(fitness_over_gen)
            self.ax3.set_ylim(min(min_beta, min_fitness), max(max_beta, max_fitness))

            self.current_frame = frame
            self.generation_label.config(text=f"Generación: {frame}")

            # Guardar el frame actual como imagen PNG en el directorio temporal
            frame_filename = os.path.join(self.temp_dir, f"frame_{frame:04d}.png")
            self.fig3.savefig(frame_filename)
            print(f"Frame guardado: {frame_filename}")

            # Si es el último frame, generar el video
            if frame == len(self.generations_betas) -1:
                print("Último frame alcanzado, generando video...")
                self.anim.event_source.stop()
                self.save_video_animation()

            return self.lines_betas + [self.line_fitness]

        try:
            self.anim = FuncAnimation(self.fig3, update,
                                      frames=range(len(self.generations_betas)),
                                      init_func=init,
                                      blit=False,
                                      interval=500,
                                      repeat=False)
        except Exception as e:
            messagebox.showerror("Error de Animación", f"No se pudo crear la animación:\n{e}")
            self.anim = None

        self.canvas3.draw()

    def save_video_animation(self):
        if self.anim is not None:
            try:
                output_file = "ga_animation.mp4"  # Nombre del archivo de video
                self.create_video_from_frames(self.temp_dir, output_file, fps=5)
                messagebox.showinfo("Éxito", f"El video se ha guardado como '{output_file}'.")
            except Exception as e:
                messagebox.showerror("Error al Guardar", f"No se pudo guardar el video:\n{e}")
        else:
            messagebox.showwarning("Animación No Disponible", "No hay animación para guardar.")

    # Función para crear video a partir de frames usando OpenCV
    def create_video_from_frames(self, temp_dir, output_file, fps=5):
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Obtener y ordenar los archivos de los frames
        frame_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".png")])
        print(f"Frames encontrados para video: {len(frame_files)}")
        if not frame_files:
            raise ValueError("No se encontraron frames en el directorio temporal.")

        # Leer el primer frame para obtener el tamaño del video
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            raise ValueError(f"No se pudo leer el frame: {frame_files[0]}")

        frame_size = (first_frame.shape[1], first_frame.shape[0])  # (width, height)
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)

        # Escribir cada frame en el video
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is None:
                print(f"No se pudo leer el frame: {frame_file}")
                continue  # Saltear frames que no se pudieron leer
            out.write(frame)

        out.release()
        print(f"Video generado exitosamente en {output_file}")

    def next_frame(self):
        # Avanzar un cuadro en la animación.
        if self.anim is not None:
            if self.current_frame < len(self.generations_betas) -1:
                self.current_frame += 1
                self.update_animation_frame(self.current_frame)
            else:
                messagebox.showinfo("Fin de la Animación", "Ya has llegado al final de la animación.")
        else:
            messagebox.showwarning("Animación No Disponible", "No hay animación para avanzar.")

    def prev_frame(self):
        # Retroceder un cuadro en la animación.
        if self.anim is not None:
            if self.current_frame > 0:
                self.current_frame -= 1
                self.update_animation_frame(self.current_frame)
            else:
                messagebox.showinfo("Inicio de la Animación", "Ya estás en el inicio de la animación.")
        else:
            messagebox.showwarning("Animación No Disponible", "No hay animación para retroceder.")

    def replay_animation(self):
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except AttributeError:
                pass

            # Reiniciar la animación
            self.current_frame = 0
            self.is_paused = False
            self.generation_label.config(text=f"Generación: {self.current_frame}")

            # Limpiar el directorio temporal antes de crear nuevos frames
            shutil.rmtree(self.temp_dir)
            self.temp_dir = tempfile.mkdtemp()
            print(f"Directorio temporal limpiado y recreado: {self.temp_dir}")

            self.plot_animation()
        else:
            messagebox.showwarning("Animación No Disponible", "No hay animación para reiniciar.")

    def update_animation_frame(self, frame):
        # Actualizar la gráfica con datos específicos de una generación
        generations = range(frame +1)
        betas_over_gen = np.array(self.generations_betas[:frame +1])
        fitness_over_gen = [1 / fit -1 for fit in self.generations_fit[:frame +1]]

        for beta_idx in range(self.num_betas):
            self.lines_betas[beta_idx].set_data(generations, betas_over_gen[:, beta_idx])
        self.line_fitness.set_data(generations, fitness_over_gen)

        # Ajustar límites del gráfico
        self.ax3.set_xlim(0, len(self.generations_betas))
        all_betas = betas_over_gen.flatten()
        min_beta = min(all_betas)
        max_beta = max(all_betas)
        min_fitness = min(fitness_over_gen)
        max_fitness = max(fitness_over_gen)
        self.ax3.set_ylim(min(min_beta, min_fitness), max(max_beta, max_fitness))

        self.current_frame = frame
        self.generation_label.config(text=f"Generación: {frame}")

        self.canvas3.draw()

    def disable_animation_controls(self):
        # Deshabilitar controles de animación (Retroceder, Avanzar, Replay)
        self.prev_button.config(state="disabled")
        self.next_button.config(state="disabled")
        self.replay_button.config(state="disabled")
        self.generation_label.config(text="Generación: 0")

    def enable_animation_controls(self):
        # Habilitar controles de animación (Retroceder, Avanzar, Replay)
        self.prev_button.config(state="normal")
        self.next_button.config(state="normal")
        self.replay_button.config(state="normal")
        self.generation_label.config(text=f"Generación: {self.current_frame}")

    def __del__(self):
        # Limpiar el directorio temporal al cerrar la aplicación
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Directorio temporal eliminado: {self.temp_dir}")
