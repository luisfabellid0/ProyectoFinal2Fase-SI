# Este cuaderno de Google Colab entrena y evalúa varios modelos de regresión
# para predecir precios de propiedades, incluyendo SVR, RandomForest y GradientBoosting.
# Utiliza bibliotecas estándar de ciencia de datos como pandas, numpy y sklearn.

from IPython import get_ipython
from IPython.display import display

# %%
# Celda 1: Importaciones de Bibliotecas
# Importamos todas las bibliotecas necesarias al principio para tener una visión general de las dependencias.
import pandas as pd
import numpy as np
import time # Para medir tiempos de ejecución

# Para Google Colab (descomentar si se necesita cargar archivos desde el entorno local)
# from google.colab import files
# import io

# Módulos de Scikit-learn para preprocesamiento y división de datos
from sklearn.model_selection import train_test_split # Para dividir datos en conjuntos de entrenamiento y prueba
from sklearn.preprocessing import StandardScaler, PolynomialFeatures # StandardScaler para escalar características, PolynomialFeatures para crear características polinómicas

# Módulos de Scikit-learn para modelos de regresión
from sklearn.svm import SVR # Máquinas de Vectores de Soporte para Regresión
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # Modelos de ensamble
from sklearn.model_selection import GridSearchCV # Para optimización de hiperparámetros usando búsqueda en cuadrícula
from sklearn.base import clone # Para clonar estimadores, útil para usar el mejor estimador de GridSearchCV en otros pasos

# Módulos de Scikit-learn para métricas de evaluación
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Métricas comunes de regresión

# Módulos para visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns # Basado en Matplotlib, proporciona una interfaz de alto nivel para gráficos estadísticos atractivos

# %%
# Celda 2: Carga de Datos
# Cargamos el dataset desde un archivo CSV. Se incluye manejo básico de errores
# en caso de que el archivo no se encuentre.
# Se asume que el archivo 'train_ranked_by_zip_price.csv' ya ha sido subido
# al entorno de Colab o está accesible en la ruta especificada.
try:
    df = pd.read_csv('train_ranked_by_zip_price.csv')
    print("Dataset loaded successfully. First 5 rows:")
    print(df.head())
    # Descomentar la siguiente línea para ver información detallada del dataset (tipos de datos, valores no nulos)
    # print("\nDataset information:")
    # df.info()
except FileNotFoundError:
    # Mensaje de error si el archivo no se encuentra, guiando al usuario.
    print("Error: 'train_ranked_by_zip_price.csv' not found. ")
    print("Please make sure the file is uploaded or the path is correct.")

# %%
# Celda 3: Definición de Características y Objetivo + Preprocesamiento Básico
# Definimos las columnas que usaremos como características (X) y la columna objetivo (y).
# Se realizan comprobaciones básicas para asegurar que las columnas existen y que la columna objetivo es numérica.
# También se manejan los valores faltantes en las columnas de características imputando con la mediana.
if 'df' in locals() or 'df' in globals():
    # Lista de nombres de las columnas que serán utilizadas como variables independientes (features).
    features = [
        'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'taxamount',
        'bathroomcnt', 'bedroomcnt', 'yearbuilt', 'latitude', 'longitude',
        'regionidzip', 'regionidcounty', 'zip_price_rank'
    ]
    # Nombre de la columna que queremos predecir (variable dependiente o objetivo).
    target_col = 'price'

    # Verificamos si la columna objetivo existe en el DataFrame.
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
    # Verificamos si la columna objetivo es numérica, ya que es un problema de regresión.
    elif not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: Target column '{target_col}' is not numeric.")
    else:
        # Eliminamos filas donde la columna objetivo tenga valores nulos.
        df.dropna(subset=[target_col], inplace=True)
        # Verificamos si todas las columnas de características existen en el DataFrame.
        missing_feature_cols = [col for col in features if col not in df.columns]
        if missing_feature_cols:
            print(f"Error: Missing feature columns: {missing_feature_cols}")
        else:
            # Creamos los DataFrames X (características) y Series y (objetivo).
            X = df[features].copy()
            y = df[target_col].copy()
            # Iteramos sobre las columnas de características para manejar valores faltantes.
            for col in X.columns:
                if X[col].isnull().any():
                    # Imputamos los valores faltantes en cada columna de características con la mediana de esa columna.
                    # print(f"Warning: Column '{col}' has {X[col].isnull().sum()} missing values. Imputing with median.") # Descomentar para detalle
                    X.loc[:, col] = X[col].fillna(X[col].median())
            print("Features (X) and target (y) defined. Missing values handled.")
else:
    # Mensaje si el DataFrame principal no fue cargado en la celda anterior.
    print("DataFrame 'df' not loaded. Run Cell 2.")

# %%
# Celda 4: División de Datos
# Dividimos los datos en conjuntos de entrenamiento y prueba.
# Esto es crucial para evaluar el rendimiento del modelo en datos no vistos.
# Un test_size de 0.2 significa que el 20% de los datos se usarán para prueba.
# random_state asegura que la división sea la misma cada vez que se ejecute el código.
if 'X' in locals() and 'y' in locals():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split: X_train: {X_train.shape}, X_test: {X_test.shape}")
else:
    # Mensaje si X y/o y no fueron definidos en la celda anterior.
    print("X and/or y not defined. Run Cell 3.")

# %%
# Celda 5: Escalado de Características (principalmente para SVR)
# Escalamos las características para los modelos que son sensibles a la escala de los datos,
# como SVR. StandardScaler normaliza los datos para que tengan media 0 y varianza 1.
# Se entrena el escalador solo con los datos de entrenamiento para evitar fuga de información del conjunto de prueba.
if 'X_train' in locals():
    scaler_svr = StandardScaler() # Creamos una instancia del escalador.
    # Entrenamos el escalador y transformamos el conjunto de entrenamiento.
    X_train_scaled_svr = scaler_svr.fit_transform(X_train)
    # Solo transformamos el conjunto de prueba usando el escalador entrenado con los datos de entrenamiento.
    X_test_scaled_svr = scaler_svr.transform(X_test)
    print("Features scaled for SVR (X_train_scaled_svr, X_test_scaled_svr).")
else:
    # Mensaje si los datos de entrenamiento no están disponibles.
    print("X_train not defined. Run Cell 4.")

# %%
# Celda 6: Función Auxiliar para Evaluación y Almacenamiento de Resultados
# Definimos una función para evaluar un modelo dado en el conjunto de prueba
# y almacenar las métricas de rendimiento (R²) y las predicciones.
model_performance = {} # Diccionario para almacenar el R² de cada modelo evaluado.
predictions_repo = {}  # Diccionario para almacenar las predicciones de cada modelo evaluado.

def evaluate_model(model_name, model, X_test_data, y_test_data, store_in_dict=True):
    """
    Evalúa un modelo de regresión dado en los datos de prueba y muestra las métricas.
    Opcionalmente, almacena el rendimiento (R²) y las predicciones.

    Args:
        model_name (str): Nombre descriptivo del modelo para la salida y almacenamiento.
        model: El objeto del modelo entrenado de Scikit-learn.
        X_test_data (array-like): Las características del conjunto de prueba.
        y_test_data (array-like): Los valores reales del objetivo del conjunto de prueba.
        store_in_dict (bool): Si es True, almacena el R² y las predicciones en los diccionarios globales.

    Returns:
        tuple: (y_pred, r2) Las predicciones y el valor R² del modelo.
    """
    print(f"\n--- Evaluación del Modelo: {model_name} ---")
    start_time = time.time() # Registramos el tiempo de inicio de la predicción.

    # Realizamos las predicciones sobre los datos de prueba.
    y_pred = model.predict(X_test_data)

    eval_time = time.time() - start_time # Calculamos el tiempo que tomó hacer las predicciones.

    # Calculamos las métricas de evaluación.
    mae = mean_absolute_error(y_test_data, y_pred) # Error Absoluto Medio
    rmse = np.sqrt(mean_squared_error(y_test_data, y_pred)) # Raíz del Error Cuadrático Medio
    r2 = r2_score(y_test_data, y_pred) # Coeficiente de Determinación R²

    # Imprimimos las métricas calculadas.
    print(f"  Tiempo de predicción: {eval_time:.2f} segundos")
    print(f"  Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"  Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"  R-squared (R²): {r2:.4f}")

    # Si se indica, almacenamos el R² y las predicciones en los diccionarios globales.
    if store_in_dict:
        model_performance[model_name] = r2
        predictions_repo[model_name] = y_pred

    # Devolvemos las predicciones y el valor R².
    return y_pred, r2

print("Función de evaluación 'evaluate_model' definida.")
print("Diccionarios 'model_performance' y 'predictions_repo' inicializados.")

# %%
# Celda 7: Modelo SVR Base
# Entrenamos un modelo SVR (Support Vector Regressor) con un conjunto inicial de hiperparámetros.
# Este modelo utiliza los datos escalados (X_train_scaled_svr, X_test_scaled_svr) ya que SVR es sensible a la escala.
if 'X_train_scaled_svr' in locals():
    print("\n--- Modelo SVR Base ---")
    # Inicializamos el modelo SVR con algunos parámetros base.
    initial_svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    print("Entrenando SVR Base...")
    start_time = time.time() # Registramos el tiempo de inicio del entrenamiento.
    # Entrenamos el modelo SVR con los datos de entrenamiento escalados.
    initial_svr_model.fit(X_train_scaled_svr, y_train)
    train_time = time.time() - start_time # Calculamos el tiempo de entrenamiento.
    print(f"SVR Base entrenado en {train_time:.2f} segundos.")
    # Evaluamos el modelo entrenado usando la función auxiliar.
    evaluate_model("SVR Base", initial_svr_model, X_test_scaled_svr, y_test)
else:
    # Mensaje si los datos escalados para SVR no están disponibles.
    print("Datos escalados para SVR (X_train_scaled_svr) no disponibles. Ejecute la celda 5.")

# %%
# Celda 8: SVR con Optimización de Hiperparámetros (GridSearchCV)
# Realizamos una búsqueda en cuadrícula (GridSearchCV) para encontrar los mejores hiperparámetros
# para el modelo SVR. GridSearchCV prueba todas las combinaciones de parámetros en la cuadrícula
# y utiliza validación cruzada para evaluar cada combinación.
if 'X_train_scaled_svr' in locals():
    print("\n--- SVR Optimizado con GridSearchCV ---")
    # Definimos la cuadrícula de hiperparámetros a probar.
    # Se incluyen comentarios para sugerir valores más amplios a probar para una búsqueda exhaustiva.
    param_grid_svr = {
        'C': [50, 200],          # Rango de valores para el parámetro de regularización C.
        'gamma': [0.1],     # Rango de valores para el parámetro gamma del kernel RBF.
        'epsilon': [0.1],    # Rango de valores para el parámetro epsilon en la función de pérdida de SVR.
        'kernel': ['rbf']         # Tipo de kernel a usar (RBF es común para SVR).
    }

    # Inicializamos GridSearchCV.
    # cv=2 o 3 para ejecución más rápida en demo/pruebas. Usar cv=5 o más para resultados más robustos.
    # n_jobs=-1 usa todos los procesadores disponibles.
    # scoring='r2' indica que usaremos el R² como métrica para seleccionar el mejor modelo.
    # verbose=1 muestra información sobre el progreso.
    grid_search_svr = GridSearchCV(estimator=SVR(), param_grid=param_grid_svr,
                                   cv=2, n_jobs=-1, scoring='r2', verbose=1)

    print("Iniciando GridSearchCV para SVR (puede tardar)...")
    start_time = time.time() # Registramos el tiempo de inicio del GridSearchCV.
    # Ejecutamos la búsqueda en cuadrícula con los datos de entrenamiento escalados.
    grid_search_svr.fit(X_train_scaled_svr, y_train)
    train_time = time.time() - start_time # Calculamos el tiempo que tomó GridSearchCV.
    print(f"GridSearchCV para SVR completado en {train_time:.2f} segundos.")

    # Obtenemos el mejor estimador encontrado por GridSearchCV.
    best_svr_model = grid_search_svr.best_estimator_
    # Imprimimos los mejores parámetros encontrados y el mejor R² de validación cruzada.
    print(f"\nMejores parámetros SVR: {grid_search_svr.best_params_}")
    print(f"Mejor R² (cross-validation) SVR: {grid_search_svr.best_score_:.4f}")

    # Evaluamos el mejor modelo SVR encontrado en el conjunto de prueba escalado.
    evaluate_model("SVR Optimizado (GridSearchCV)", best_svr_model, X_test_scaled_svr, y_test)
else:
    # Mensaje si los datos escalados para SVR no están disponibles.
    print("Datos escalados para SVR (X_train_scaled_svr) no disponibles.")

# %%
# Celda 9: SVR Optimizado con Características Polinómicas
# Exploramos si añadir características polinómicas mejora el rendimiento del SVR optimizado.
# PolynomialFeatures crea nuevas características elevando las características existentes a una potencia
# y creando términos de interacción entre ellas. Estas nuevas características también deben ser escaladas.
if 'X_train' in locals() and 'best_svr_model' in locals():
    print("\n--- SVR Optimizado con Características Polinómicas ---")
    # Inicializamos el transformador de características polinómicas.
    # degree=2 crea características hasta el grado 2 (incluyendo términos x^2, y^2 y x*y).
    # include_bias=False evita añadir una columna de unos.
    # interaction_only=False incluye tanto términos polinómicos como de interacción.
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)

    print("Creando características polinómicas...")
    # Transformamos los datos de entrenamiento y prueba. Es importante usar X_train (no escalado)
    # para ajustar el transformador PolynomialFeatures y luego transformarlo.
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    print(f"Nueva forma de X_train con PolyFeatures: {X_train_poly.shape}")

    # Escalar las nuevas características polinómicas. Se necesita un nuevo escalador
    # porque el número de características ha cambiado.
    scaler_poly = StandardScaler()
    X_train_poly_scaled = scaler_poly.fit_transform(X_train_poly)
    X_test_poly_scaled = scaler_poly.transform(X_test_poly)
    print("Características polinómicas escaladas.")

    # Clonamos el mejor modelo SVR encontrado previamente con GridSearchCV.
    svr_poly_model = clone(best_svr_model)

    print("Entrenando SVR Optimizado con PolyFeatures...")
    start_time = time.time() # Registramos el tiempo de inicio del entrenamiento.
    # Entrenamos el modelo SVR clonado con los datos polinómicos escalados.
    svr_poly_model.fit(X_train_poly_scaled, y_train)
    train_time = time.time() - start_time # Calculamos el tiempo de entrenamiento.
    print(f"SVR con PolyFeatures entrenado en {train_time:.2f} segundos.")

    # Evaluamos el modelo SVR con características polinómicas escaladas.
    evaluate_model("SVR Optimizado con PolyFeatures", svr_poly_model, X_test_poly_scaled, y_test)
elif 'best_svr_model' not in locals():
    # Mensaje si el mejor modelo SVR de GridSearchCV no fue encontrado.
    print("El modelo 'best_svr_model' de GridSearchCV no está disponible. Ejecute la celda 8.")
else:
    # Mensaje si los datos de entrenamiento originales no están disponibles.
    print("X_train no disponible.")

# %%
# Celda 10: Modelo RandomForestRegressor
# Entrenamos un modelo RandomForestRegressor. Los modelos basados en árboles generalmente
# no requieren escalado de características, por lo que usamos los datos originales (no escalados).
# RandomForest es un modelo de ensamble que construye múltiples árboles de decisión y promedia sus predicciones.
if 'X_train' in locals():
    print("\n--- Modelo RandomForestRegressor ---")
    # Inicializamos el modelo RandomForestRegressor con algunos parámetros de ejemplo.
    # Se incluyen comentarios para sugerir afinamiento de hiperparámetros.
    rf_model = RandomForestRegressor(n_estimators=100, # Número de árboles en el bosque.
                                     random_state=42, # Semilla para reproducibilidad.
                                     n_jobs=-1,        # Usa todos los núcleos de CPU disponibles.
                                     max_depth=20,       # Profundidad máxima de los árboles.
                                     min_samples_split=10, # Mínimo de muestras requeridas para dividir un nodo interno.
                                     min_samples_leaf=5) # Mínimo de muestras requeridas para ser un nodo hoja.
    print("Entrenando RandomForestRegressor...")
    start_time = time.time() # Registramos el tiempo de inicio del entrenamiento.
    # Entrenamos el modelo con los datos de entrenamiento originales (no escalados).
    rf_model.fit(X_train, y_train)
    train_time = time.time() - start_time # Calculamos el tiempo de entrenamiento.
    print(f"RandomForestRegressor entrenado en {train_time:.2f} segundos.")

    # Evaluamos el modelo con los datos de prueba originales (no escalados).
    evaluate_model("RandomForestRegressor", rf_model, X_test, y_test)
    print("Nota: Para mejores resultados, los hiperparámetros de RandomForest deberían ser afinados (ej. GridSearchCV).")
else:
    # Mensaje si los datos de entrenamiento no están disponibles.
    print("X_train no disponible.")

# %%
# Celda 11: Modelo GradientBoostingRegressor
# Entrenamos un modelo GradientBoostingRegressor, otro modelo de ensamble basado en árboles.
# GradientBoosting construye árboles secuencialmente, corrigiendo los errores de los árboles anteriores.
# Al igual que RandomForest, generalmente no requiere escalado de características.
if 'X_train' in locals():
    print("\n--- Modelo GradientBoostingRegressor ---")
    # Inicializamos el modelo GradientBoostingRegressor con algunos parámetros de ejemplo.
    # Se incluyen comentarios para sugerir afinamiento de hiperparámetros.
    gb_model = GradientBoostingRegressor(n_estimators=150,      # Número de etapas de boosting (árboles).
                                         learning_rate=0.05,   # Contribución de cada árbol (reduce el efecto de cada árbol).
                                         max_depth=4,          # Profundidad máxima de los estimadores individuales.
                                         random_state=42, # Semilla para reproducibilidad.
                                         subsample=0.8,        # Fracción de muestras utilizadas para entrenar cada árbol.
                                         min_samples_leaf=5) # Mínimo de muestras requeridas para ser un nodo hoja.
    print("Entrenando GradientBoostingRegressor...")
    start_time = time.time() # Registramos el tiempo de inicio del entrenamiento.
    # Entrenamos el modelo con los datos de entrenamiento originales (no escalados).
    gb_model.fit(X_train, y_train)
    train_time = time.time() - start_time # Calculamos el tiempo de entrenamiento.
    print(f"GradientBoostingRegressor entrenado en {train_time:.2f} segundos.")

    # Evaluamos el modelo con los datos de prueba originales (no escalados).
    evaluate_model("GradientBoostingRegressor", gb_model, X_test, y_test)
    print("Nota: Para mejores resultados, los hiperparámetros de GradientBoosting deberían ser afinados.")
else:
    # Mensaje si los datos de entrenamiento no están disponibles.
    print("X_train no disponible.")

# %%
# Celda 12: Comparación de Modelos y Análisis de Errores del Mejor Modelo
# Comparamos el rendimiento (R²) de todos los modelos entrenados y evaluados.
# Identificamos el modelo con el mejor R² en el conjunto de prueba.
# Finalmente, generamos un gráfico de residuos para el mejor modelo para visualizar la distribución de los errores.
if model_performance:
    print("\n--- Comparación Final de Rendimiento de Modelos (R²) ---")
    best_model_name_final = None
    best_r2_final = -float('inf') # Inicializamos el mejor R² con un valor muy bajo.

    # Ordenamos los modelos por su rendimiento R² de forma descendente para presentar los resultados.
    sorted_performance = sorted(model_performance.items(), key=lambda item: item[1], reverse=True)

    # Iteramos sobre los modelos ordenados, imprimimos su rendimiento y encontramos el mejor.
    for name, r2_val in sorted_performance:
        print(f"  {name}: R² = {r2_val:.4f}")
        # La condición siempre será verdadera para el primer elemento después de ordenar.
        if r2_val > best_r2_final:
            best_r2_final = r2_val
            best_model_name_final = name

    # Si se encontró un mejor modelo, lo anunciamos y procedemos al análisis de residuos.
    if best_model_name_final:
        print(f"\nMejor modelo global basado en R²: {best_model_name_final} (R²: {best_r2_final:.4f})")

        # Gráfico de Residuos para el mejor modelo.
        # Un gráfico de residuos muestra la diferencia entre los valores reales y los predichos
        # en función de los valores predichos. Idealmente, los residuos deben estar distribuidos
        # aleatoriamente alrededor de cero.
        if best_model_name_final in predictions_repo:
            # Obtenemos las predicciones del mejor modelo.
            y_pred_for_residuals = predictions_repo[best_model_name_final]
            # Calculamos los residuos.
            residuals = y_test - y_pred_for_residuals

            # Creamos el gráfico de dispersión de residuos.
            plt.figure(figsize=(10, 6)) # Define el tamaño de la figura.
            sns.scatterplot(x=y_pred_for_residuals, y=residuals, alpha=0.5) # Gráfico de dispersión con transparencia.
            plt.axhline(0, color='red', linestyle='--') # Línea horizontal en y=0 para referencia.
            plt.xlabel(f"Valores Predichos ({best_model_name_final})") # Etiqueta del eje X.
            plt.ylabel("Residuos (Actual - Predicho)") # Etiqueta del eje Y.
            plt.title(f"Gráfico de Residuos del Modelo: {best_model_name_final}") # Título del gráfico.
            plt.grid(True) # Muestra una cuadrícula.
            plt.show() # Muestra el gráfico.
        else:
            # Mensaje si las predicciones del mejor modelo no se almacenaron.
            print(f"No se encontraron predicciones almacenadas para el mejor modelo ({best_model_name_final}) para el gráfico de residuos.")
    else:
        # Mensaje si no se pudo determinar el mejor modelo (por ejemplo, si model_performance estaba vacío).
        print("No se pudo determinar el mejor modelo.")
else:
    # Mensaje si no hay resultados de modelos para comparar.
    print("\nNo hay resultados de modelos para comparar. Ejecute las celdas de entrenamiento y evaluación.")

print("\n--- Fin del Proceso de Modelado y Evaluación ---")