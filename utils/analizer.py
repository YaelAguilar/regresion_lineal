import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def recomendar_funcion_activacion():
    # Configurar ventana para seleccionar archivo
    Tk().withdraw()
    archivo = askopenfilename(title="Seleccionar archivo CSV", 
                            filetypes=[("Archivos CSV", "*.csv")])
    
    if not archivo:
        print("No se seleccionó ningún archivo.")
        return
    
    try:
        # Leer el archivo CSV
        df = pd.read_csv(archivo, delimiter=';')
        
        # Verificar estructura del archivo
        if len(df.columns) < 3 or df.columns[-1] != 'Y':
            raise ValueError("Formato de archivo incorrecto")
            
        # Obtener la variable objetivo
        y = df.iloc[:, -1]
        
        # Análisis de la variable Y
        if y.dtype == 'object':
            # Clasificación categórica
            num_clases = y.nunique()
            if num_clases == 2:
                recomendacion = "Sigmoid (clasificación binaria)"
            else:
                recomendacion = "Softmax (clasificación multiclase)"
        else:
            # Análisis para variables numéricas
            valores_unicos = y.nunique()
            
            if valores_unicos == 2:
                # Verificar si son 0 y 1
                if set(y.unique()) == {0, 1}:
                    recomendacion = "Sigmoid (clasificación binaria)"
                else:
                    recomendacion = "Sigmoid (clasificación binaria con valores no estándar)"
            else:
                # Verificar si es discreto con múltiples clases
                if y.apply(float.is_integer).all() and valores_unicos <= 10:
                    recomendacion = "Softmax (clasificación multiclase)"
                else:
                    recomendacion = "Lineal (problema de regresión)"
        
        print("\nRecomendación de función de activación:")
        print(f"- {recomendacion}")
        print("\nConsideraciones adicionales:")
        print("- Para problemas de regresión, usar pérdida MSE")
        print("- Para clasificación binaria, usar pérdida Binaria Crossentropy")
        print("- Para multiclase, usar Categorical Crossentropy")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Verifica que el archivo tenga el formato correcto:")
        print("- Columnas: id;X1;...;Xn;Y")
        print("- Delimitador ;")
        print("- Encabezados en primera fila")

if __name__ == "__main__":
    recomendar_funcion_activacion()