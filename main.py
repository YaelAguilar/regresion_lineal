import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from utils.data_loader import load_data
from utils.linear_model import LinearRegression

class RegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Regresión Lineal")
        self.data = None
        self.setup_ui()

    def setup_ui(self):
        # Frame de controles
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Botón para cargar archivo
        ttk.Button(control_frame, text="Cargar CSV", command=self.load_file).pack(side=tk.LEFT)

        # Dropdown para selección de columnas (REFERENCIAS DIRECTAS)
        self.feature_var = tk.StringVar()
        self.target_var = tk.StringVar()
        ttk.Label(control_frame, text="Feature:").pack(side=tk.LEFT, padx=5)
        self.feature_combobox = ttk.Combobox(control_frame, textvariable=self.feature_var, state='readonly')
        self.feature_combobox.pack(side=tk.LEFT)
        ttk.Label(control_frame, text="Target:").pack(side=tk.LEFT, padx=5)
        self.target_combobox = ttk.Combobox(control_frame, textvariable=self.target_var, state='readonly')
        self.target_combobox.pack(side=tk.LEFT)
        
        # Botón para entrenar modelo
        ttk.Button(control_frame, text="Generar Modelo", command=self.train_model).pack(side=tk.LEFT, padx=10)

        # Área de gráficos
        self.figure = plt.Figure(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.data = load_data(file_path)
            self.update_comboboxes()

    def update_comboboxes(self):
        columns = self.data.columns.tolist()
        # Actualizar directamente los combobox usando las referencias guardadas (CORRECTO)
        self.feature_combobox['values'] = columns
        self.target_combobox['values'] = columns

    def train_model(self):
        if self.data is not None:
            X = self.data[self.feature_var.get()].values
            y = self.data[self.target_var.get()].values
            
            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)

            # Actualizar gráfico
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.scatter(X, y, color='blue', label='Datos')
            ax.plot(X, predictions, color='red', label='Regresión')
            ax.set_xlabel(self.feature_var.get())
            ax.set_ylabel(self.target_var.get())
            ax.legend()
            self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RegressionApp(root)
    root.mainloop()