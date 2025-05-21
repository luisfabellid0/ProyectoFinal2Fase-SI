import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
# CELDA AUXILIAR (Configuración de Kaggle)
# Esta celda no cambia respecto a la versión anterior,
# asume que kaggle.json está en /content/kaggle.json
print("Configurando Kaggle CLI...")
# Crear el directorio de Kaggle si no existe
!mkdir -p ~/.kaggle
# Mover kaggle.json de /content/ al directorio de Kaggle
!cp /content/kaggle.json ~/.kaggle/
# Establecer permisos adecuados para el archivo kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

print("Archivo kaggle.json configurado desde /content/kaggle.json.")

# CELDA 2 (Modificada): PARTE 1 - Carga y Lectura del Dataset (Solo Zillow Properties)
import pandas as pd
import numpy as np
import os

print("--- PARTE 1: CARGA Y LECTURA DEL DATASET (SOLO properties_2016.csv) ---")
df = None # Inicializar df (que será df_properties)

# Nombre del archivo de propiedades
properties_file_name = 'properties_2016.csv'
# Dataset de Kaggle: zillow-prize-1
kaggle_dataset_name = 'zillow-prize-1'
properties_file_path = properties_file_name # Asumimos que estará en el directorio actual después de la descarga/descompresión

if not os.path.exists(properties_file_name):
    print(f"Descargando '{properties_file_name}' desde Kaggle (esto puede tardar unos minutos)...")

    try:
        # El siguiente comando asume que kaggle.json está configurado.
        print(f"Intentando descargar '{properties_file_name}.zip'...")
        !kaggle competitions download -c {kaggle_dataset_name} -f {properties_file_name}.zip --force

        if os.path.exists(f'{properties_file_name}.zip'):
             print(f"Descomprimiendo '{properties_file_name}.zip'...")
             !unzip -o {properties_file_name}.zip # -o para sobrescribir sin preguntar
             print("Descompresión completada.")
             if not os.path.exists(properties_file_name):
                 print(f"Error: El archivo '{properties_file_name}' no se encontró después de la descompresión.")
                 properties_file_path = None
        elif os.path.exists(properties_file_name): # Si se descargó directamente el CSV (sin .zip)
            print(f"'{properties_file_name}' descargado directamente.")
        else: # Si no se encontró el .zip ni el .csv directamente
             print(f"No se encontró '{properties_file_name}.zip'. Intentando descargar '{properties_file_name}' directamente.")
             !kaggle competitions download -c {kaggle_dataset_name} -f {properties_file_name} --force
             if not os.path.exists(properties_file_name):
                 print(f"Error: '{properties_file_name}' no se pudo descargar. Verifica el nombre del archivo y tu API de Kaggle.")
                 properties_file_path = None

        if properties_file_path and os.path.exists(properties_file_path):
            print(f"'{properties_file_name}' disponible exitosamente.")
        elif not properties_file_path: # Si ya se marcó como None
            pass # El mensaje de error ya se imprimió
        else: # Si properties_file_path no es None pero el archivo no existe
            print(f"Error: '{properties_file_name}' no se pudo obtener. Verifica el proceso de descarga/descompresión.")
            properties_file_path = None

    except Exception as e:
        print(f"Ocurrió un error durante la descarga/descompresión: {e}")
        properties_file_path = None
else:
    print(f"'{properties_file_name}' ya existe en el directorio.")
    # properties_file_path ya está asignado a properties_file_name


# --- Cargar el dataframe de propiedades ---
try:
    if properties_file_path and os.path.exists(properties_file_path):
        print(f"Cargando '{properties_file_path}'...")

        # Definición de tipos de datos y columnas a seleccionar para properties_2016.csv
        dtype_properties = {
            'parcelid': np.int32,
            'bathroomcnt': np.float32,
            'bedroomcnt': np.float32,
            'calculatedfinishedsquarefeet': np.float32,
            'yearbuilt': np.float32, # CAMBIADO de float16 a float32
            'taxvaluedollarcnt': np.float32,
            'taxamount': np.float32,
            'latitude': np.float32,
            'longitude': np.float32,
            'regionidzip': np.float32, # CAMBIADO de float16 a float32
            'regionidcounty': np.float32 # CAMBIADO de float16 a float32
        }
        selected_property_features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                                      'yearbuilt', 'taxvaluedollarcnt', 'taxamount', 'latitude', 'longitude',
                                      'regionidzip', 'regionidcounty']

        # Cargar df_properties usando solo las columnas seleccionadas
        df_properties = pd.read_csv(properties_file_path, usecols=selected_property_features, dtype=dtype_properties)
        print(f"'{properties_file_path}' cargado. Forma: {df_properties.shape}")

        # El dataset principal es ahora df_properties
        df = df_properties

        print(f"Tamaño del dataset ('{properties_file_name}'): {df.shape[0]} filas, {df.shape[1]} columnas")

        print("\n--- Primeras 5 filas de TODAS las columnas del dataset ('properties_2016.csv'): ---")
        print(df.head())

        print("\n--- Primeras 5 filas de las COLUMNAS SELECCIONADAS ('selected_property_features') del dataset: ---")
        # Nota: Dado que df (df_properties) se cargó utilizando usecols=selected_property_features,
        # este output será idéntico al anterior, ya que todas las columnas del DataFrame son las seleccionadas.
        cols_to_display = [col for col in selected_property_features if col in df.columns]
        if cols_to_display: # Esta comprobación es por robustez
            print(df[cols_to_display].head())
        else:
            # Esto no debería ocurrir si df se cargó correctamente con selected_property_features
            print("Advertencia: No se pudieron encontrar las columnas de 'selected_property_features' en el DataFrame 'df'.")

    else:
        print(f"Error: No se pudo encontrar o acceder al archivo de propiedades en la ruta: {properties_file_path}")
        df = None # Asegurar que df es None si el archivo de propiedades no se carga

except Exception as e:
    print(f"Ocurrió un error general al cargar los datos de propiedades: {e}")
    if 'df' not in locals() or df is None: # Si df no se creó o falló su creación
         print("El DataFrame 'df' (proveniente de properties_2016.csv) no pudo ser creado.")
import pandas as pd
import numpy as np
import os

# --- PASO 0: CONFIGURACIÓN DE KAGGLE ---
print("Configurando Kaggle CLI...")
# Crear el directorio de Kaggle si no existe
!mkdir -p ~/.kaggle
# Mover kaggle.json de /content/ al directorio de Kaggle
# Asumiendo que 'kaggle.json' ya está en '/content/kaggle.json'
!cp /content/kaggle.json ~/.kaggle/
# Establecer permisos adecuados para el archivo kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
print("Archivo kaggle.json configurado desde /content/kaggle.json.")
print("-" * 50)

# --- PASO 1: Definiciones de archivos y parámetros ---
print("--- PASO 1: Definiciones de archivos y parámetros ---")
original_properties_file_name = 'properties_2016.csv'
cleaned_properties_file_name = 'properties_2016_cleaned_yearbuilt.csv' # Nombre del nuevo CSV limpio
kaggle_dataset_name = 'zillow-prize-1'

# Definición de tipos de datos y columnas a seleccionar para properties_2016.csv
dtype_properties = {
    'parcelid': np.int32,
    'bathroomcnt': np.float32,
    'bedroomcnt': np.float32,
    'calculatedfinishedsquarefeet': np.float32,
    'yearbuilt': np.float32, # Se mantiene como float para manejar NaNs antes de la limpieza
    'taxvaluedollarcnt': np.float32,
    'taxamount': np.float32,
    'latitude': np.float32,
    'longitude': np.float32,
    'regionidzip': np.float32,
    'regionidcounty': np.float32
}
selected_property_features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                              'yearbuilt', 'taxvaluedollarcnt', 'taxamount', 'latitude', 'longitude',
                              'regionidzip', 'regionidcounty']
print(f"Archivo original a procesar: {original_properties_file_name}")
print(f"Archivo limpio a generar y leer: {cleaned_properties_file_name}")
print("-" * 50)

# --- PASO 2: Descargar el archivo original de propiedades si no existe ---
print(f"--- PASO 2: Verificación y descarga de '{original_properties_file_name}' ---")
original_file_path = original_properties_file_name
if not os.path.exists(original_properties_file_name):
    print(f"'{original_properties_file_name}' no encontrado. Descargando desde Kaggle...")
    try:
        print(f"Intentando descargar '{original_properties_file_name}.zip'...")
        # Asegúrate de que la competición y el nombre del archivo son correctos en Kaggle
        !kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name}.zip --force

        if os.path.exists(f'{original_properties_file_name}.zip'):
             print(f"Descomprimiendo '{original_properties_file_name}.zip'...")
             !unzip -o {original_properties_file_name}.zip # -o para sobrescribir
             print("Descompresión completada.")
             if not os.path.exists(original_properties_file_name):
                 print(f"Error: El archivo '{original_properties_file_name}' no se encontró después de la descompresión.")
                 original_file_path = None # Marcar que el archivo no está disponible
        elif os.path.exists(original_properties_file_name): # Si se descargó directamente el CSV
            print(f"'{original_properties_file_name}' descargado directamente.")
        else: # Si no se encontró .zip ni el .csv
             print(f"No se encontró '{original_properties_file_name}.zip'. Intentando descargar '{original_properties_file_name}' directamente.")
             !kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name} --force
             if not os.path.exists(original_properties_file_name):
                 print(f"Error: '{original_properties_file_name}' no se pudo descargar.")
                 original_file_path = None # Marcar que el archivo no está disponible

        if original_file_path and os.path.exists(original_file_path):
            print(f"'{original_properties_file_name}' disponible exitosamente.")
        elif not original_file_path: # Si ya se marcó como None
            pass
        else:
            print(f"Error: '{original_properties_file_name}' no se pudo obtener. Verificar proceso de descarga.")
            original_file_path = None

    except Exception as e:
        print(f"Ocurrió un error durante la descarga/descompresión de '{original_properties_file_name}': {e}")
        original_file_path = None
else:
    print(f"'{original_properties_file_name}' ya existe en el directorio.")
print("-" * 50)

# --- PASO 3: Cargar, limpiar y guardar el dataset ---
df_cleaned_intermediate = None # DataFrame intermedio después de la limpieza
success_step_3 = False

if original_file_path and os.path.exists(original_file_path):
    print(f"--- PASO 3: Carga, limpieza de '{original_properties_file_name}' y guardado ---")
    try:
        print(f"Cargando '{original_file_path}' para limpieza...")
        df_temp = pd.read_csv(original_file_path, usecols=selected_property_features, dtype=dtype_properties)
        print(f"'{original_file_path}' cargado. Forma original: {df_temp.shape}")

        # Limpieza: eliminar filas donde 'yearbuilt' es NaN
        initial_rows = len(df_temp)
        df_cleaned_intermediate = df_temp.dropna(subset=['yearbuilt']) # Clave de la limpieza
        cleaned_rows = len(df_cleaned_intermediate)
        rows_removed = initial_rows - cleaned_rows

        print(f"Limpieza de 'yearbuilt':")
        print(f"  - Filas iniciales: {initial_rows}")
        print(f"  - Filas con 'yearbuilt' NaN eliminadas: {rows_removed}")
        print(f"  - Filas restantes (limpias): {cleaned_rows}")
        print(f"Forma después de la limpieza: {df_cleaned_intermediate.shape}")

        # Guardar el DataFrame limpio en un nuevo archivo CSV
        df_cleaned_intermediate.to_csv(cleaned_properties_file_name, index=False)
        print(f"DataFrame limpio guardado exitosamente en '{cleaned_properties_file_name}'.")
        success_step_3 = True

    except Exception as e:
        print(f"Ocurrió un error al cargar, limpiar o guardar los datos de '{original_file_path}': {e}")
else:
    print(f"No se puede proceder con PASO 3 ya que '{original_properties_file_name}' no está disponible o no se pudo descargar.")
print("-" * 50)

# --- PASO 4: Cargar el nuevo CSV limpio y mostrar información ---
df = None # DataFrame final que se leerá del archivo limpio

if success_step_3 and os.path.exists(cleaned_properties_file_name):
    print(f"--- PASO 4: Carga y visualización del dataset limpio '{cleaned_properties_file_name}' ---")
    try:
        print(f"Cargando el dataset limpio '{cleaned_properties_file_name}'...")
        # Al leer el CSV que guardamos, ya contiene las columnas y tipos correctos.
        # Re-aplicar dtype_properties es una buena práctica para asegurar la consistencia.
        df = pd.read_csv(cleaned_properties_file_name, dtype=dtype_properties)
        print(f"'{cleaned_properties_file_name}' cargado exitosamente.")
        print(f"Forma del dataset final limpio: {df.shape[0]} filas, {df.shape[1]} columnas.")

        print("\n📊 Primeras 5 filas de TODAS las columnas del dataset limpio:")
        print(df.head())

        print("\n📋 Primeras 5 filas de las COLUMNAS SELECCIONADAS ('selected_property_features') del dataset limpio:")
        # Nota: Este output será idéntico al anterior, ya que el CSV limpio
        # se generó usando solo estas columnas.
        cols_to_display = [col for col in selected_property_features if col in df.columns]
        if cols_to_display:
            print(df[cols_to_display].head())
        else:
            # Esto no debería ocurrir si los pasos anteriores fueron exitosos.
            print("Advertencia: Las columnas de 'selected_property_features' no se encontraron en el DataFrame limpio.")

    except Exception as e:
        print(f"Ocurrió un error al cargar el archivo CSV limpio '{cleaned_properties_file_name}': {e}")
        df = None # Asegurar que df es None si hay error
else:
    if not success_step_3:
        print(f"No se puede proceder con PASO 4 porque el PASO 3 (limpieza y guardado) no fue exitoso.")
    elif not os.path.exists(cleaned_properties_file_name):
         print(f"Error en PASO 4: El archivo CSV limpio '{cleaned_properties_file_name}' no fue encontrado. "
               "Esto indica un problema en el guardado del PASO 3.")
    df = None # Asegurar que df es None

# --- Resumen final del estado del DataFrame ---
if df is not None:
    print(f"\n✅ Proceso completado. El DataFrame 'df' contiene los datos limpios de '{cleaned_properties_file_name}'.")
else:
    print(f"\n❌ Proceso no completado. El DataFrame 'df' no pudo ser creado/cargado con los datos limpios.")

import pandas as pd
import numpy as np
import os

# --- PASO 0: CONFIGURACIÓN DE KAGGLE ---
print("Configurando Kaggle CLI...")
# Crear el directorio de Kaggle si no existe
!mkdir -p ~/.kaggle
# Mover kaggle.json de /content/ al directorio de Kaggle
# Asumiendo que 'kaggle.json' ya está en '/content/kaggle.json'
!cp /content/kaggle.json ~/.kaggle/
# Establecer permisos adecuados para el archivo kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
print("Archivo kaggle.json configurado desde /content/kaggle.json.")
print("-" * 50)

# --- PASO 1: Definiciones de archivos y parámetros ---
print("--- PASO 1: Definiciones de archivos y parámetros ---")
original_properties_file_name = 'properties_2016.csv'
cleaned_properties_file_name = 'properties_2016_cleaned_yearbuilt.csv' # Nombre del nuevo CSV limpio
kaggle_dataset_name = 'zillow-prize-1'

# Definición de tipos de datos y columnas a seleccionar para properties_2016.csv
dtype_properties = {
    'parcelid': np.int32,
    'bathroomcnt': np.float32,
    'bedroomcnt': np.float32,
    'calculatedfinishedsquarefeet': np.float32,
    'yearbuilt': np.float32, # Se mantiene como float para manejar NaNs antes de la limpieza
    'taxvaluedollarcnt': np.float32,
    'taxamount': np.float32,
    'latitude': np.float32,
    'longitude': np.float32,
    'regionidzip': np.float32,
    'regionidcounty': np.float32
}
selected_property_features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                              'yearbuilt', 'taxvaluedollarcnt', 'taxamount', 'latitude', 'longitude',
                              'regionidzip', 'regionidcounty']
print(f"Archivo original a procesar: {original_properties_file_name}")
print(f"Archivo limpio a generar y leer: {cleaned_properties_file_name}")
print("-" * 50)

# --- PASO 2: Descargar el archivo original de propiedades si no existe ---
print(f"--- PASO 2: Verificación y descarga de '{original_properties_file_name}' ---")
original_file_path = original_properties_file_name
if not os.path.exists(original_properties_file_name):
    print(f"'{original_properties_file_name}' no encontrado. Descargando desde Kaggle...")
    try:
        print(f"Intentando descargar '{original_properties_file_name}.zip'...")
        # Asegúrate de que la competición y el nombre del archivo son correctos en Kaggle
        !kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name}.zip --force

        if os.path.exists(f'{original_properties_file_name}.zip'):
             print(f"Descomprimiendo '{original_properties_file_name}.zip'...")
             !unzip -o {original_properties_file_name}.zip # -o para sobrescribir
             print("Descompresión completada.")
             if not os.path.exists(original_properties_file_name):
                 print(f"Error: El archivo '{original_properties_file_name}' no se encontró después de la descompresión.")
                 original_file_path = None # Marcar que el archivo no está disponible
        elif os.path.exists(original_properties_file_name): # Si se descargó directamente el CSV
            print(f"'{original_properties_file_name}' descargado directamente.")
        else: # Si no se encontró .zip ni el .csv
             print(f"No se encontró '{original_properties_file_name}.zip'. Intentando descargar '{original_properties_file_name}' directamente.")
             !kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name} --force
             if not os.path.exists(original_properties_file_name):
                 print(f"Error: '{original_properties_file_name}' no se pudo descargar.")
                 original_file_path = None # Marcar que el archivo no está disponible

        if original_file_path and os.path.exists(original_file_path):
            print(f"'{original_properties_file_name}' disponible exitosamente.")
        elif not original_file_path: # Si ya se marcó como None
            pass
        else:
            print(f"Error: '{original_properties_file_name}' no se pudo obtener. Verificar proceso de descarga.")
            original_file_path = None

    except Exception as e:
        print(f"Ocurrió un error durante la descarga/descompresión de '{original_properties_file_name}': {e}")
        original_file_path = None
else:
    print(f"'{original_properties_file_name}' ya existe en el directorio.")
print("-" * 50)

# --- PASO 3: Cargar, limpiar y guardar el dataset ---
df_cleaned_intermediate = None # DataFrame intermedio después de la limpieza
success_step_3 = False

if original_file_path and os.path.exists(original_file_path):
    print(f"--- PASO 3: Carga, limpieza de '{original_properties_file_name}' y guardado ---")
    try:
        print(f"Cargando '{original_file_path}' para limpieza...")
        df_temp = pd.read_csv(original_file_path, usecols=selected_property_features, dtype=dtype_properties)
        print(f"'{original_file_path}' cargado. Forma original: {df_temp.shape}")

        # Limpieza: eliminar filas donde 'yearbuilt' es NaN
        initial_rows = len(df_temp)
        df_cleaned_intermediate = df_temp.dropna(subset=['yearbuilt']) # Clave de la limpieza
        cleaned_rows = len(df_cleaned_intermediate)
        rows_removed = initial_rows - cleaned_rows

        print(f"Limpieza de 'yearbuilt':")
        print(f"  - Filas iniciales: {initial_rows}")
        print(f"  - Filas con 'yearbuilt' NaN eliminadas: {rows_removed}")
        print(f"  - Filas restantes (limpias): {cleaned_rows}")
        print(f"Forma después de la limpieza: {df_cleaned_intermediate.shape}")

        # Guardar el DataFrame limpio en un nuevo archivo CSV
        df_cleaned_intermediate.to_csv(cleaned_properties_file_name, index=False)
        print(f"DataFrame limpio guardado exitosamente en '{cleaned_properties_file_name}'.")
        success_step_3 = True

    except Exception as e:
        print(f"Ocurrió un error al cargar, limpiar o guardar los datos de '{original_file_path}': {e}")
else:
    print(f"No se puede proceder con PASO 3 ya que '{original_properties_file_name}' no está disponible o no se pudo descargar.")
print("-" * 50)

# --- PASO 4: Cargar el nuevo CSV limpio y mostrar información ---
df = None # DataFrame final que se leerá del archivo limpio

if success_step_3 and os.path.exists(cleaned_properties_file_name):
    print(f"--- PASO 4: Carga y visualización del dataset limpio '{cleaned_properties_file_name}' ---")
    try:
        print(f"Cargando el dataset limpio '{cleaned_properties_file_name}'...")
        # Al leer el CSV que guardamos, ya contiene las columnas y tipos correctos.
        # Re-aplicar dtype_properties es una buena práctica para asegurar la consistencia.
        df = pd.read_csv(cleaned_properties_file_name, dtype=dtype_properties)
        print(f"'{cleaned_properties_file_name}' cargado exitosamente.")
        print(f"Forma del dataset final limpio: {df.shape[0]} filas, {df.shape[1]} columnas.")

        print("\n📊 Primeras 5 filas de TODAS las columnas del dataset limpio:")
        print(df.head())

        print("\n📋 Primeras 5 filas de las COLUMNAS SELECCIONADAS ('selected_property_features') del dataset limpio:")
        # Nota: Este output será idéntico al anterior, ya que el CSV limpio
        # se generó usando solo estas columnas.
        cols_to_display = [col for col in selected_property_features if col in df.columns]
        if cols_to_display:
            print(df[cols_to_display].head())
        else:
            # Esto no debería ocurrir si los pasos anteriores fueron exitosos.
            print("Advertencia: Las columnas de 'selected_property_features' no se encontraron en el DataFrame limpio.")

    except Exception as e:
        print(f"Ocurrió un error al cargar el archivo CSV limpio '{cleaned_properties_file_name}': {e}")
        df = None # Asegurar que df es None si hay error
else:
    if not success_step_3:
        print(f"No se puede proceder con PASO 4 porque el PASO 3 (limpieza y guardado) no fue exitoso.")
    elif not os.path.exists(cleaned_properties_file_name):
         print(f"Error en PASO 4: El archivo CSV limpio '{cleaned_properties_file_name}' no fue encontrado. "
               "Esto indica un problema en el guardado del PASO 3.")
    df = None # Asegurar que df es None

# --- Resumen final del estado del DataFrame ---
if df is not None:
    print(f"\n✅ Proceso completado. El DataFrame 'df' contiene los datos limpios de '{cleaned_properties_file_name}'.")
else:
    print(f"\n❌ Proceso no completado. El DataFrame 'df' no pudo ser creado/cargado con los datos limpios.")

import pandas as pd
import numpy as np
import os

# --- PASO 0: CONFIGURACIÓN DE KAGGLE ---
print("Configurando Kaggle CLI...")
# Crear el directorio de Kaggle si no existe
!mkdir -p ~/.kaggle
# Mover kaggle.json de /content/ al directorio de Kaggle
# Asumiendo que 'kaggle.json' ya está en '/content/kaggle.json'
!cp /content/kaggle.json ~/.kaggle/
# Establecer permisos adecuados para el archivo kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
print("Archivo kaggle.json configurado desde /content/kaggle.json.")
print("-" * 50)

# --- PASO 1: Definiciones de archivos y parámetros ---
print("--- PASO 1: Definiciones de archivos y parámetros ---")
original_properties_file_name = 'properties_2016.csv'
# Nombre del nuevo CSV limpio actualizado para reflejar múltiples limpiezas
cleaned_properties_file_name = 'properties_2016_cleaned.csv'
kaggle_dataset_name = 'zillow-prize-1'

# Definición de tipos de datos y columnas a seleccionar para properties_2016.csv
dtype_properties = {
    'parcelid': np.int32,
    'bathroomcnt': np.float32,
    'bedroomcnt': np.float32,
    'calculatedfinishedsquarefeet': np.float32,
    'yearbuilt': np.float32,
    'taxvaluedollarcnt': np.float32,
    'taxamount': np.float32,
    'latitude': np.float32,
    'longitude': np.float32,
    'regionidzip': np.float32,
    'regionidcounty': np.float32
}
selected_property_features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                              'yearbuilt', 'taxvaluedollarcnt', 'taxamount', 'latitude', 'longitude',
                              'regionidzip', 'regionidcounty']
print(f"Archivo original a procesar: {original_properties_file_name}")
print(f"Archivo limpio a generar y leer: {cleaned_properties_file_name}")
print("-" * 50)

# --- PASO 2: Descargar el archivo original de propiedades si no existe ---
print(f"--- PASO 2: Verificación y descarga de '{original_properties_file_name}' ---")
original_file_path = original_properties_file_name
if not os.path.exists(original_properties_file_name):
    print(f"'{original_properties_file_name}' no encontrado. Descargando desde Kaggle...")
    try:
        print(f"Intentando descargar '{original_properties_file_name}.zip'...")
        !kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name}.zip --force

        if os.path.exists(f'{original_properties_file_name}.zip'):
             print(f"Descomprimiendo '{original_properties_file_name}.zip'...")
             !unzip -o {original_properties_file_name}.zip
             print("Descompresión completada.")
             if not os.path.exists(original_properties_file_name):
                 print(f"Error: El archivo '{original_properties_file_name}' no se encontró después de la descompresión.")
                 original_file_path = None
        elif os.path.exists(original_properties_file_name):
            print(f"'{original_properties_file_name}' descargado directamente.")
        else:
             print(f"No se encontró '{original_properties_file_name}.zip'. Intentando descargar '{original_properties_file_name}' directamente.")
             !kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name} --force
             if not os.path.exists(original_properties_file_name):
                 print(f"Error: '{original_properties_file_name}' no se pudo descargar.")
                 original_file_path = None

        if original_file_path and os.path.exists(original_file_path):
            print(f"'{original_properties_file_name}' disponible exitosamente.")
        elif not original_file_path:
            pass
        else:
            print(f"Error: '{original_properties_file_name}' no se pudo obtener. Verificar proceso de descarga.")
            original_file_path = None

    except Exception as e:
        print(f"Ocurrió un error durante la descarga/descompresión de '{original_properties_file_name}': {e}")
        original_file_path = None
else:
    print(f"'{original_properties_file_name}' ya existe en el directorio.")
print("-" * 50)

# --- PASO 3: Cargar, limpiar y guardar el dataset ---
df_cleaned_intermediate = None
success_step_3 = False

if original_file_path and os.path.exists(original_file_path):
    print(f"--- PASO 3: Carga, limpieza de '{original_properties_file_name}' y guardado ---")
    try:
        print(f"Cargando '{original_file_path}' para limpieza...")
        df_temp = pd.read_csv(original_file_path, usecols=selected_property_features, dtype=dtype_properties)
        print(f"'{original_file_path}' cargado. Forma original: {df_temp.shape}")

        current_rows = len(df_temp)
        print(f"Filas antes de cualquier limpieza: {current_rows}")

        # Limpieza 1: eliminar filas donde 'yearbuilt' es NaN
        print("\nIniciando limpieza de 'yearbuilt'...")
        df_cleaned_step1 = df_temp.dropna(subset=['yearbuilt'])
        rows_after_step1 = len(df_cleaned_step1)
        print(f"  - Filas después de eliminar NaN en 'yearbuilt': {rows_after_step1} (Eliminadas: {current_rows - rows_after_step1})")
        current_rows = rows_after_step1

        # Limpieza 2: eliminar filas donde 'bathroomcnt' es 0
        print("\nIniciando limpieza de 'bathroomcnt'...")
        df_cleaned_step2 = df_cleaned_step1[df_cleaned_step1['bathroomcnt'] != 0]
        rows_after_step2 = len(df_cleaned_step2)
        print(f"  - Filas después de eliminar registros con 'bathroomcnt' == 0: {rows_after_step2} (Eliminadas: {current_rows - rows_after_step2})")
        current_rows = rows_after_step2

        # Limpieza 3: eliminar filas donde 'bedroomcnt' es 0
        print("\nIniciando limpieza de 'bedroomcnt'...")
        df_cleaned_step3 = df_cleaned_step2[df_cleaned_step2['bedroomcnt'] != 0]
        rows_after_step3 = len(df_cleaned_step3)
        print(f"  - Filas después de eliminar registros con 'bedroomcnt' == 0: {rows_after_step3} (Eliminadas: {current_rows - rows_after_step3})")

        df_cleaned_intermediate = df_cleaned_step3 # DataFrame final después de todas las limpiezas

        print(f"\nForma final después de todas las limpiezas: {df_cleaned_intermediate.shape}")

        # Guardar el DataFrame limpio en un nuevo archivo CSV
        df_cleaned_intermediate.to_csv(cleaned_properties_file_name, index=False)
        print(f"DataFrame limpio guardado exitosamente en '{cleaned_properties_file_name}'.")
        success_step_3 = True

    except Exception as e:
        print(f"Ocurrió un error al cargar, limpiar o guardar los datos de '{original_file_path}': {e}")
else:
    print(f"No se puede proceder con PASO 3 ya que '{original_properties_file_name}' no está disponible o no se pudo descargar.")
print("-" * 50)

# --- PASO 4: Cargar el nuevo CSV limpio y mostrar información ---
df = None

if success_step_3 and os.path.exists(cleaned_properties_file_name):
    print(f"--- PASO 4: Carga y visualización del dataset limpio '{cleaned_properties_file_name}' ---")
    try:
        print(f"Cargando el dataset limpio '{cleaned_properties_file_name}'...")
        df = pd.read_csv(cleaned_properties_file_name, dtype=dtype_properties)
        print(f"'{cleaned_properties_file_name}' cargado exitosamente.")
        print(f"Forma del dataset final limpio: {df.shape[0]} filas, {df.shape[1]} columnas.")

        print("\n📊 Primeras 5 filas de TODAS las columnas del dataset limpio:")
        print(df.head())

        print("\n📋 Primeras 5 filas de las COLUMNAS SELECCIONADAS ('selected_property_features') del dataset limpio:")
        cols_to_display = [col for col in selected_property_features if col in df.columns]
        if cols_to_display:
            print(df[cols_to_display].head())
        else:
            print("Advertencia: Las columnas de 'selected_property_features' no se encontraron en el DataFrame limpio.")

    except Exception as e:
        print(f"Ocurrió un error al cargar el archivo CSV limpio '{cleaned_properties_file_name}': {e}")
        df = None
else:
    if not success_step_3:
        print(f"No se puede proceder con PASO 4 porque el PASO 3 (limpieza y guardado) no fue exitoso.")
    elif not os.path.exists(cleaned_properties_file_name):
         print(f"Error en PASO 4: El archivo CSV limpio '{cleaned_properties_file_name}' no fue encontrado. "
               "Esto indica un problema en el guardado del PASO 3.")
    df = None

# --- Resumen final del estado del DataFrame ---
if df is not None:
    print(f"\n✅ Proceso completado. El DataFrame 'df' contiene los datos limpios de '{cleaned_properties_file_name}'.")
else:
    print(f"\n❌ Proceso no completado. El DataFrame 'df' no pudo ser creado/cargado con los datos limpios.")
import pandas as pd
import numpy as np
import os

# --- PASO 0: CONFIGURACIÓN DE KAGGLE ---
print("Configurando Kaggle CLI...")
# Crear el directorio de Kaggle si no existe
!mkdir -p ~/.kaggle
# Mover kaggle.json de /content/ al directorio de Kaggle
# Asumiendo que 'kaggle.json' ya está en '/content/kaggle.json'
if os.path.exists('/content/kaggle.json'):
    !cp /content/kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    print("Archivo kaggle.json configurado desde /content/kaggle.json.")
else:
    print("Advertencia: /content/kaggle.json no encontrado. La descarga desde Kaggle podría fallar si no está configurado manualmente.")
print("-" * 50)

# --- PASO 1: Definiciones de archivos y parámetros ---
print("--- PASO 1: Definiciones de archivos y parámetros ---")
original_properties_file_name = 'properties_2016.csv'
cleaned_properties_file_name = 'properties_2016_cleaned.csv'
kaggle_dataset_name = 'zillow-prize-1'

# Definición de tipos de datos y columnas a seleccionar
dtype_properties = {
    'parcelid': np.int32,
    'bathroomcnt': np.float32,
    'bedroomcnt': np.float32,
    'calculatedfinishedsquarefeet': np.float32,
    'yearbuilt': np.float32,
    'taxvaluedollarcnt': np.float32,
    'taxamount': np.float32,
    'latitude': np.float32,
    'longitude': np.float32,
    'regionidzip': np.float32, # Columna clave para el conteo
    'regionidcounty': np.float32
}
selected_property_features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                              'yearbuilt', 'taxvaluedollarcnt', 'taxamount', 'latitude', 'longitude',
                              'regionidzip', 'regionidcounty']
print(f"Archivo original a procesar: {original_properties_file_name}")
print(f"Archivo limpio (base para el conteo): {cleaned_properties_file_name}")
print("-" * 50)

# --- PASO 2: Descargar el archivo original si es necesario ---
# (Esta sección se ejecutará si el archivo limpio no existe y necesitamos generar el original primero)
original_file_path = original_properties_file_name
if not os.path.exists(cleaned_properties_file_name) and not os.path.exists(original_properties_file_name):
    print(f"--- PASO 2: '{cleaned_properties_file_name}' no existe. Verificando y descargando '{original_properties_file_name}' ---")
    print(f"'{original_properties_file_name}' no encontrado. Descargando desde Kaggle...")
    try:
        print(f"Intentando descargar '{original_properties_file_name}.zip'...")
        !kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name}.zip --force

        if os.path.exists(f'{original_properties_file_name}.zip'):
             print(f"Descomprimiendo '{original_properties_file_name}.zip'...")
             !unzip -o {original_properties_file_name}.zip
             print("Descompresión completada.")
             if not os.path.exists(original_properties_file_name):
                 print(f"Error: El archivo '{original_properties_file_name}' no se encontró después de la descompresión.")
                 original_file_path = None
        elif os.path.exists(original_properties_file_name):
            print(f"'{original_properties_file_name}' descargado directamente.")
        else:
             print(f"No se encontró '{original_properties_file_name}.zip'. Intentando descargar '{original_properties_file_name}' directamente.")
             !kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name} --force
             if not os.path.exists(original_properties_file_name):
                 print(f"Error: '{original_properties_file_name}' no se pudo descargar.")
                 original_file_path = None

        if original_file_path and os.path.exists(original_file_path):
            print(f"'{original_properties_file_name}' disponible exitosamente.")
        elif not original_file_path:
            pass
        else:
            print(f"Error: '{original_properties_file_name}' no se pudo obtener. Verificar proceso de descarga.")
            original_file_path = None

    except Exception as e:
        print(f"Ocurrió un error durante la descarga/descompresión de '{original_properties_file_name}': {e}")
        original_file_path = None
elif os.path.exists(original_properties_file_name):
     print(f"--- PASO 2: '{original_properties_file_name}' ya existe. No se necesita descarga. ---")
else:
    pass # El archivo limpio ya existe o el original no es necesario si el limpio sí.
print("-" * 50)

# --- PASO 3: Cargar, limpiar y guardar el dataset si el archivo limpio no existe ---
success_step_3 = False
if not os.path.exists(cleaned_properties_file_name):
    print(f"--- PASO 3: '{cleaned_properties_file_name}' no existe. Generándolo... ---")
    if original_file_path and os.path.exists(original_file_path):
        try:
            print(f"Cargando '{original_file_path}' para limpieza...")
            df_temp = pd.read_csv(original_file_path, usecols=selected_property_features, dtype=dtype_properties)
            print(f"'{original_file_path}' cargado. Forma original: {df_temp.shape}")

            current_rows = len(df_temp)
            print(f"Filas antes de cualquier limpieza: {current_rows}")

            print("\nIniciando limpieza de 'yearbuilt'...")
            df_cleaned_step1 = df_temp.dropna(subset=['yearbuilt'])
            rows_after_step1 = len(df_cleaned_step1)
            print(f"  - Filas después de eliminar NaN en 'yearbuilt': {rows_after_step1} (Eliminadas: {current_rows - rows_after_step1})")
            current_rows = rows_after_step1

            print("\nIniciando limpieza de 'bathroomcnt'...")
            df_cleaned_step2 = df_cleaned_step1[df_cleaned_step1['bathroomcnt'] != 0]
            rows_after_step2 = len(df_cleaned_step2)
            print(f"  - Filas después de eliminar registros con 'bathroomcnt' == 0: {rows_after_step2} (Eliminadas: {current_rows - rows_after_step2})")
            current_rows = rows_after_step2

            print("\nIniciando limpieza de 'bedroomcnt'...")
            df_cleaned_step3 = df_cleaned_step2[df_cleaned_step2['bedroomcnt'] != 0]
            rows_after_step3 = len(df_cleaned_step3)
            print(f"  - Filas después de eliminar registros con 'bedroomcnt' == 0: {rows_after_step3} (Eliminadas: {current_rows - rows_after_step3})")

            df_cleaned_intermediate = df_cleaned_step3

            print(f"\nForma final después de todas las limpiezas: {df_cleaned_intermediate.shape}")
            df_cleaned_intermediate.to_csv(cleaned_properties_file_name, index=False)
            print(f"DataFrame limpio guardado exitosamente en '{cleaned_properties_file_name}'.")
            success_step_3 = True

        except Exception as e:
            print(f"Ocurrió un error al cargar, limpiar o guardar los datos de '{original_file_path}': {e}")
    else:
        print(f"No se puede generar '{cleaned_properties_file_name}' ya que '{original_properties_file_name}' no está disponible.")
else:
    print(f"--- PASO 3: El archivo limpio '{cleaned_properties_file_name}' ya existe. No se necesita regeneración. ---")
    success_step_3 = True # Consideramos exitoso si el archivo ya existe
print("-" * 50)

# --- PASO 4: Cargar el CSV limpio y realizar el conteo por regionidzip ---
df = None
if success_step_3 and os.path.exists(cleaned_properties_file_name):
    print(f"--- PASO 4: Carga de '{cleaned_properties_file_name}' y conteo por 'regionidzip' ---")
    try:
        print(f"Cargando el dataset limpio '{cleaned_properties_file_name}'...")
        df = pd.read_csv(cleaned_properties_file_name, dtype=dtype_properties)
        print(f"'{cleaned_properties_file_name}' cargado exitosamente.")
        print(f"Forma del dataset: {df.shape[0]} filas, {df.shape[1]} columnas.")

        if 'regionidzip' in df.columns:
            print("\n📊 Conteo de propiedades por 'regionidzip':")

            # Contar valores, convertir NaN a un string para que aparezca si existen, luego ordenar por zip
            # Si regionidzip es float y tiene NaNs, value_counts() los omite por defecto.
            # Para incluirlos como una categoría específica o asegurar que todos son tratados:
            # df_for_count = df.copy()
            # df_for_count['regionidzip'] = df_for_count['regionidzip'].fillna('Desconocido')
            # region_counts = df_for_count['regionidzip'].value_counts().sort_index()

            # Opción simple: contar y ordenar. Los NaNs en regionidzip serán omitidos por value_counts().
            region_counts = df['regionidzip'].value_counts().sort_index()

            region_counts_df = region_counts.reset_index()
            region_counts_df.columns = ['regionidzip', 'numero_de_propiedades']

            print(region_counts_df)
            print(f"\nTotal de 'regionidzip' distintos (excluyendo NaN): {len(region_counts_df)}")

            # Si quieres ver si hay NaNs en regionidzip:
            nan_in_zip = df['regionidzip'].isna().sum()
            if nan_in_zip > 0:
                print(f"Número de propiedades con 'regionidzip' NaN (no incluidas en la tabla anterior): {nan_in_zip}")

        else:
            print("Error: La columna 'regionidzip' no se encuentra en el DataFrame cargado.")

    except Exception as e:
        print(f"Ocurrió un error al cargar el archivo CSV limpio '{cleaned_properties_file_name}' o al contar: {e}")
        df = None
else:
    if not os.path.exists(cleaned_properties_file_name):
         print(f"Error en PASO 4: El archivo CSV limpio '{cleaned_properties_file_name}' no fue encontrado y no pudo ser generado.")
    df = None

# --- Resumen final del estado del DataFrame ---
if df is not None:
    print(f"\n✅ Proceso completado. El DataFrame 'df' contiene los datos limpios de '{cleaned_properties_file_name}' y se ha mostrado el conteo.")
else:
    print(f"\n❌ Proceso no completado. El DataFrame 'df' no pudo ser cargado/procesado.")
import pandas as pd
import numpy as np
import os

# --- PASO 0: CONFIGURACIÓN DE KAGGLE ---
print("Configurando Kaggle CLI...")
if os.path.exists('/content/kaggle.json'):
    !mkdir -p ~/.kaggle
    !cp /content/kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    print("Archivo kaggle.json configurado desde /content/kaggle.json.")
else:
    print("Advertencia: /content/kaggle.json no encontrado. La descarga desde Kaggle podría fallar si no está configurado manualmente.")
print("-" * 50)

# --- PASO 1: Definiciones de archivos y parámetros ---
print("--- PASO 1: Definiciones de archivos y parámetros ---")
original_properties_file_name = 'properties_2016.csv'
cleaned_properties_file_name = 'properties_2016_cleaned.csv'
train_file_name = 'train_2016_v2.csv'
filtered_train_file_name = 'train_2016_v2_filtered_by_cleaned_properties.csv' # Nombre del archivo de train filtrado
kaggle_dataset_name = 'zillow-prize-1'

# Dtypes y features para properties
dtype_properties = {
    'parcelid': np.int32, 'bathroomcnt': np.float32, 'bedroomcnt': np.float32,
    'calculatedfinishedsquarefeet': np.float32, 'yearbuilt': np.float32,
    'taxvaluedollarcnt': np.float32, 'taxamount': np.float32,
    'latitude': np.float32, 'longitude': np.float32,
    'regionidzip': np.float32, 'regionidcounty': np.float32
}
selected_property_features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                              'yearbuilt', 'taxvaluedollarcnt', 'taxamount', 'latitude', 'longitude',
                              'regionidzip', 'regionidcounty']

# Dtypes para train (importante para parcelid)
dtype_train = {
    'parcelid': np.int32
}
print(f"Archivo de propiedades original: {original_properties_file_name}")
print(f"Archivo de propiedades limpio: {cleaned_properties_file_name}")
print(f"Archivo de transacciones original: {train_file_name}")
print(f"Archivo de transacciones filtrado a generar: {filtered_train_file_name}")
print("-" * 50)

# --- PASO 2: Asegurar que properties_2016_cleaned.csv esté disponible (Generarlo si no existe) ---
success_generating_cleaned_properties = False
if not os.path.exists(cleaned_properties_file_name):
    print(f"--- PASO 2a: '{cleaned_properties_file_name}' no existe. Intentando generarlo... ---")
    original_file_path = original_properties_file_name
    if not os.path.exists(original_properties_file_name):
        print(f"'{original_properties_file_name}' no encontrado. Descargando desde Kaggle...")
        try:
            !kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name}.zip --force
            if os.path.exists(f'{original_properties_file_name}.zip'):
                !unzip -o {original_properties_file_name}.zip
                print(f"'{original_properties_file_name}' descargado y descomprimido.")
            elif os.path.exists(original_properties_file_name):
                 print(f"'{original_properties_file_name}' descargado directamente.")
            else:
                raise Exception(f"No se pudo encontrar {original_properties_file_name} ni su .zip después de intentar descargar.")
        except Exception as e:
            print(f"Error descargando '{original_properties_file_name}': {e}")
            original_file_path = None

    if original_file_path and os.path.exists(original_file_path):
        try:
            print(f"Cargando '{original_file_path}' para limpieza...")
            df_temp_props = pd.read_csv(original_file_path, usecols=selected_property_features, dtype=dtype_properties)
            current_rows = len(df_temp_props)
            df_temp_props = df_temp_props.dropna(subset=['yearbuilt'])
            df_temp_props = df_temp_props[df_temp_props['bathroomcnt'] != 0]
            df_temp_props = df_temp_props[df_temp_props['bedroomcnt'] != 0]
            print(f"Limpieza de propiedades completada. Filas restantes: {len(df_temp_props)} de {current_rows}")
            df_temp_props.to_csv(cleaned_properties_file_name, index=False)
            print(f"'{cleaned_properties_file_name}' generado y guardado.")
            success_generating_cleaned_properties = True
        except Exception as e:
            print(f"Error generando '{cleaned_properties_file_name}': {e}")
    else:
        print(f"No se pudo generar '{cleaned_properties_file_name}' porque '{original_properties_file_name}' no está disponible.")
else:
    print(f"--- PASO 2a: '{cleaned_properties_file_name}' ya existe. No se necesita regeneración. ---")
    success_generating_cleaned_properties = True
print("-" * 50)

# --- PASO 3: Asegurar que train_2016_v2.csv esté disponible ---
success_train_file_available = False
if not os.path.exists(train_file_name):
    print(f"--- PASO 3: '{train_file_name}' no encontrado. Descargando desde Kaggle... ---")
    try:
        # El archivo train_2016_v2.csv usualmente viene en un .zip llamado train_2016_v2.csv.zip o similar
        # o directamente. Vamos a intentar descargar el archivo directamente.
        # Para Zillow, es train_2016_v2.csv
        !kaggle competitions download -c {kaggle_dataset_name} -f {train_file_name} --force
        if os.path.exists(train_file_name):
            print(f"'{train_file_name}' descargado exitosamente.")
            success_train_file_available = True
        # Check for a zip if direct file not found (though for train_2016_v2.csv it's usually direct)
        elif os.path.exists(f'{train_file_name}.zip'):
            print(f"Descomprimiendo '{train_file_name}.zip'...")
            !unzip -o {train_file_name}.zip
            if os.path.exists(train_file_name):
                 print(f"'{train_file_name}' descomprimido exitosamente.")
                 success_train_file_available = True
            else:
                 print(f"Error: '{train_file_name}' no se encontró después de descomprimir el zip.")
        else:
            print(f"Error: '{train_file_name}' no se pudo descargar ni encontrar como zip.")
    except Exception as e:
        print(f"Ocurrió un error durante la descarga de '{train_file_name}': {e}")
else:
    print(f"--- PASO 3: '{train_file_name}' ya existe. No se necesita descarga. ---")
    success_train_file_available = True
print("-" * 50)

# --- PASO 4: Cargar DataFrames y Realizar Filtrado ---
df_filtered_train = None
if success_generating_cleaned_properties and success_train_file_available:
    print(f"--- PASO 4: Cargando DataFrames y aplicando filtro a '{train_file_name}' ---")
    try:
        print(f"Cargando '{cleaned_properties_file_name}'...")
        df_cleaned_properties = pd.read_csv(cleaned_properties_file_name, usecols=['parcelid'], dtype={'parcelid': np.int32})
        valid_parcelids = set(df_cleaned_properties['parcelid'])
        print(f"Se encontraron {len(valid_parcelids)} parcelids únicos en '{cleaned_properties_file_name}'.")

        print(f"Cargando '{train_file_name}'...")
        # Cargar todas las columnas de train_2016_v2.csv, parsear fechas
        df_train = pd.read_csv(train_file_name, dtype=dtype_train, parse_dates=['transactiondate'])
        print(f"Forma original de '{train_file_name}': {df_train.shape}")

        # Filtrar df_train
        original_train_rows = len(df_train)
        df_filtered_train = df_train[df_train['parcelid'].isin(valid_parcelids)]
        filtered_train_rows = len(df_filtered_train)
        rows_removed = original_train_rows - filtered_train_rows

        print(f"\nFiltrado de '{train_file_name}':")
        print(f"  - Filas originales: {original_train_rows}")
        print(f"  - Filas después del filtro (parcelid en propiedades limpias): {filtered_train_rows}")
        print(f"  - Filas eliminadas: {rows_removed}")

        if filtered_train_rows > 0:
            print(f"\nGuardando '{train_file_name}' filtrado en '{filtered_train_file_name}'...")
            df_filtered_train.to_csv(filtered_train_file_name, index=False)
            print("Guardado exitoso.")
        else:
            print("No quedaron filas en el archivo de transacciones después del filtrado.")


    except Exception as e:
        print(f"Ocurrió un error durante la carga o el filtrado: {e}")
else:
    print("No se puede proceder con el PASO 4 porque uno o ambos archivos necesarios no están disponibles/generados.")
print("-" * 50)

# --- PASO 5: Mostrar cabeza del DataFrame de transacciones filtrado ---
if df_filtered_train is not None and not df_filtered_train.empty:
    print(f"--- PASO 5: Primeras 5 filas del DataFrame de transacciones filtrado ('{filtered_train_file_name}') ---")
    print(df_filtered_train.head())
elif df_filtered_train is not None and df_filtered_train.empty:
    print("--- PASO 5: El DataFrame de transacciones filtrado está vacío. ---")
else:
    print("--- PASO 5: No se pudo generar el DataFrame de transacciones filtrado. ---")

# --- Resumen Final ---
if df_filtered_train is not None:
    print(f"\n✅ Proceso completado. El DataFrame de transacciones filtrado tiene {len(df_filtered_train)} filas.")
    if os.path.exists(filtered_train_file_name) and len(df_filtered_train) > 0 :
         print(f"Los datos filtrados se han guardado en '{filtered_train_file_name}'.")
    elif len(df_filtered_train) == 0:
         print(f"No se guardó ningún archivo porque no quedaron datos después del filtro.")

else:
    print(f"\n❌ Proceso no completado. No se pudo generar el DataFrame de transacciones filtrado.")
import pandas as pd
import numpy as np
import os

# --- PASO 0: CONFIGURACIÓN DE KAGGLE ---
print("Configurando Kaggle CLI...")
if os.path.exists('/content/kaggle.json'):
    !mkdir -p ~/.kaggle
    !cp /content/kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    print("Archivo kaggle.json configurado desde /content/kaggle.json.")
else:
    print("Advertencia: /content/kaggle.json no encontrado. La descarga desde Kaggle podría fallar si no está configurado manualmente.")
print("-" * 50)

# --- PASO 1: Definiciones de archivos y parámetros ---
print("--- PASO 1: Definiciones de archivos y parámetros ---")
original_properties_file_name = 'properties_2016.csv'
# Archivo de propiedades después de la limpieza inicial (yearbuilt, bathroomcnt, bedroomcnt)
initial_cleaned_properties_file_name = 'properties_2016_initial_cleaned.csv'
original_train_file_name = 'train_2016_v2.csv'
# Archivo final después de unir transacciones con propiedades y aplicar TODAS las limpiezas
final_cleaned_merged_train_file_name = 'train_final_cleaned_merged.csv'
kaggle_dataset_name = 'zillow-prize-1'

# Dtypes y features para properties
dtype_properties = {
    'parcelid': np.int32, 'bathroomcnt': np.float32, 'bedroomcnt': np.float32,
    'calculatedfinishedsquarefeet': np.float32, 'yearbuilt': np.float32,
    'taxvaluedollarcnt': np.float32, 'taxamount': np.float32,
    'latitude': np.float32, 'longitude': np.float32,
    'regionidzip': np.float32, 'regionidcounty': np.float32
}
selected_property_features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                              'yearbuilt', 'taxvaluedollarcnt', 'taxamount', 'latitude', 'longitude',
                              'regionidzip', 'regionidcounty'] # Asegurarse que estas columnas están

# Dtypes para train (importante para parcelid)
dtype_train = {
    'parcelid': np.int32
}
print(f"Propiedades original: {original_properties_file_name}")
print(f"Propiedades con limpieza inicial: {initial_cleaned_properties_file_name}")
print(f"Transacciones original: {original_train_file_name}")
print(f"Archivo final limpio y unido a generar: {final_cleaned_merged_train_file_name}")
print("-" * 50)

# --- PASO 2: Generar/Cargar initial_cleaned_properties_file_name ---
success_initial_cleaned_properties = False
if not os.path.exists(initial_cleaned_properties_file_name):
    print(f"--- PASO 2a: '{initial_cleaned_properties_file_name}' no existe. Intentando generarlo... ---")
    if not os.path.exists(original_properties_file_name):
        print(f"'{original_properties_file_name}' no encontrado. Descargando...")
        try:
            !kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name}.zip --force
            if os.path.exists(f'{original_properties_file_name}.zip'):
                !unzip -o {original_properties_file_name}.zip
            if not os.path.exists(original_properties_file_name): # Check if direct CSV downloaded
                 !kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name} --force
            if not os.path.exists(original_properties_file_name):
                raise Exception(f"No se pudo encontrar {original_properties_file_name} después de intentar descargar.")
            print(f"'{original_properties_file_name}' descargado.")
        except Exception as e:
            print(f"Error descargando '{original_properties_file_name}': {e}")

    if os.path.exists(original_properties_file_name):
        try:
            print(f"Cargando '{original_properties_file_name}' para limpieza inicial...")
            df_props = pd.read_csv(original_properties_file_name, usecols=selected_property_features, dtype=dtype_properties)
            rows_before = len(df_props)
            df_props = df_props.dropna(subset=['yearbuilt'])
            df_props = df_props[df_props['bathroomcnt'] != 0]
            df_props = df_props[df_props['bedroomcnt'] != 0]
            print(f"Limpieza inicial de propiedades completada. Filas restantes: {len(df_props)} de {rows_before}")
            df_props.to_csv(initial_cleaned_properties_file_name, index=False)
            print(f"'{initial_cleaned_properties_file_name}' generado y guardado.")
            success_initial_cleaned_properties = True
        except Exception as e:
            print(f"Error generando '{initial_cleaned_properties_file_name}': {e}")
    else:
        print(f"No se pudo generar '{initial_cleaned_properties_file_name}' porque '{original_properties_file_name}' no está disponible.")
else:
    print(f"--- PASO 2a: '{initial_cleaned_properties_file_name}' ya existe. ---")
    success_initial_cleaned_properties = True
print("-" * 50)

# --- PASO 3: Asegurar que original_train_file_name esté disponible ---
success_train_file_available = False
if not os.path.exists(original_train_file_name):
    print(f"--- PASO 3: '{original_train_file_name}' no encontrado. Descargando... ---")
    try:
        !kaggle competitions download -c {kaggle_dataset_name} -f {original_train_file_name} --force
        if os.path.exists(original_train_file_name):
            print(f"'{original_train_file_name}' descargado.")
            success_train_file_available = True
        elif os.path.exists(f'{original_train_file_name}.zip'): # Check for zip
            !unzip -o {original_train_file_name}.zip
            if os.path.exists(original_train_file_name):
                print(f"'{original_train_file_name}' descomprimido.")
                success_train_file_available = True
            else:
                 raise Exception(f"'{original_train_file_name}' no encontrado después de descomprimir.")
        else:
            raise Exception(f"'{original_train_file_name}' no descargado ni encontrado como zip.")
    except Exception as e:
        print(f"Error descargando '{original_train_file_name}': {e}")
else:
    print(f"--- PASO 3: '{original_train_file_name}' ya existe. ---")
    success_train_file_available = True
print("-" * 50)

# --- PASO 4: Filtrar transacciones, unir con propiedades limpias ---
df_merged = None
if success_initial_cleaned_properties and success_train_file_available:
    print(f"--- PASO 4: Filtrando transacciones y uniendo con propiedades limpias ---")
    try:
        print(f"Cargando '{initial_cleaned_properties_file_name}'...")
        df_props_cleaned = pd.read_csv(initial_cleaned_properties_file_name, dtype=dtype_properties)
        # Asegurarnos que solo usamos las columnas seleccionadas, especialmente para el merge
        df_props_cleaned = df_props_cleaned[selected_property_features]
        valid_parcelids = set(df_props_cleaned['parcelid'])
        print(f"ParcelIDs válidos de propiedades limpias: {len(valid_parcelids)}")

        print(f"Cargando '{original_train_file_name}'...")
        df_train_orig = pd.read_csv(original_train_file_name, dtype=dtype_train, parse_dates=['transactiondate'])
        print(f"Forma original de transacciones: {df_train_orig.shape}")

        # Filtrar transacciones por parcelids válidos
        df_train_ids_filtered = df_train_orig[df_train_orig['parcelid'].isin(valid_parcelids)]
        print(f"Forma de transacciones después de filtrar por ParcelID: {df_train_ids_filtered.shape}")

        # Unir (Merge) transacciones filtradas con datos de propiedades limpias
        print("Uniendo transacciones filtradas con datos de propiedades...")
        df_merged = pd.merge(df_train_ids_filtered, df_props_cleaned, on='parcelid', how='inner')
        print(f"Forma después de unir transacciones y propiedades: {df_merged.shape}")
        if df_merged.empty:
            print("Advertencia: El DataFrame unido está vacío. Verifique los ParcelIDs y el proceso de limpieza/filtrado.")

    except Exception as e:
        print(f"Error en PASO 4 (filtrado y unión): {e}")
else:
    print("No se puede proceder con PASO 4 debido a que faltan archivos base (propiedades limpias o transacciones).")
print("-" * 50)

# --- PASO 5: Limpieza final de NaNs en el DataFrame Unido ---
df_final_cleaned = None
if df_merged is not None and not df_merged.empty:
    print(f"--- PASO 5: Aplicando limpieza final de NaNs al DataFrame unido ---")
    try:
        print(f"Forma antes de la limpieza final de NaNs: {df_merged.shape}")

        columns_to_clean_nans = ['regionidzip', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet']
        # Verificar que las columnas existen antes de intentar usarlas en dropna
        missing_cols = [col for col in columns_to_clean_nans if col not in df_merged.columns]
        if missing_cols:
            print(f"Advertencia: Las siguientes columnas para limpieza de NaN no se encontraron en el DataFrame unido: {missing_cols}")
            # Podrías decidir detenerte o continuar sin limpiar esas columnas. Por ahora, continuaremos con las que sí existen.
            columns_to_clean_nans = [col for col in columns_to_clean_nans if col in df_merged.columns]

        if columns_to_clean_nans:
            df_final_cleaned = df_merged.dropna(subset=columns_to_clean_nans)
            rows_removed = len(df_merged) - len(df_final_cleaned)
            print(f"Filas eliminadas por NaNs en {columns_to_clean_nans}: {rows_removed}")
            print(f"Forma después de la limpieza final de NaNs: {df_final_cleaned.shape}")
        else:
            print("No se especificaron columnas válidas para la limpieza de NaNs o no estaban presentes. Se omite este paso.")
            df_final_cleaned = df_merged # Continuar con el df_merged si no hay nada que limpiar

        if not df_final_cleaned.empty:
            df_final_cleaned.to_csv(final_cleaned_merged_train_file_name, index=False)
            print(f"DataFrame final limpio y unido guardado en '{final_cleaned_merged_train_file_name}'.")
        else:
            print("El DataFrame está vacío después de la limpieza final de NaNs. No se guardó ningún archivo.")

    except Exception as e:
        print(f"Error en PASO 5 (limpieza final de NaNs): {e}")
elif df_merged is not None and df_merged.empty:
    print("El DataFrame unido (df_merged) estaba vacío antes de la limpieza final de NaNs. No se realiza PASO 5.")
else:
    print("No se puede proceder con PASO 5 porque el DataFrame unido (df_merged) no se generó correctamente.")
print("-" * 50)

# --- PASO 6: Mostrar cabeza del DataFrame final ---
if df_final_cleaned is not None and not df_final_cleaned.empty:
    print(f"--- PASO 6: Primeras 5 filas del DataFrame final limpio y unido ('{final_cleaned_merged_train_file_name}') ---")
    print(df_final_cleaned.head())
elif df_final_cleaned is not None and df_final_cleaned.empty:
     print(f"--- PASO 6: El DataFrame final ('{final_cleaned_merged_train_file_name}') está vacío. ---")
else:
    print(f"--- PASO 6: No se pudo generar el DataFrame final ('{final_cleaned_merged_train_file_name}'). ---")

# --- Resumen Final ---
if df_final_cleaned is not None and not df_final_cleaned.empty:
    print(f"\n✅ Proceso completado. El DataFrame final tiene {len(df_final_cleaned)} filas.")
    print(f"Los datos finales se han guardado en '{final_cleaned_merged_train_file_name}'.")
elif df_final_cleaned is not None and df_final_cleaned.empty:
     print(f"\n⚠️ Proceso completado, pero el DataFrame final está vacío después de todas las limpiezas.")
else:
    print(f"\n❌ Proceso no completado. No se pudo generar el DataFrame final.")
import pandas as pd
import numpy as np
import os
import subprocess # Para ejecutar comandos de shell de forma más robusta

# Función auxiliar para ejecutar comandos de shell e imprimir salida/error
def run_shell_command(command):
    print(f"Ejecutando: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Error (stderr): {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar comando: {e}")
        if e.stdout:
            print(f"Stdout del error: {e.stdout}")
        if e.stderr:
            print(f"Stderr del error: {e.stderr}")
        raise # Re-lanzar la excepción para detener si es crítico

# --- PASO 0: CONFIGURACIÓN DE KAGGLE ---
print("Configurando Kaggle CLI...")
if os.path.exists('/content/kaggle.json'):
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    run_shell_command(f"cp /content/kaggle.json {os.path.expanduser('~/.kaggle/')}")
    run_shell_command(f"chmod 600 {os.path.expanduser('~/.kaggle/kaggle.json')}")
    print("Archivo kaggle.json configurado desde /content/kaggle.json.")
else:
    print("Advertencia: /content/kaggle.json no encontrado. La descarga desde Kaggle podría fallar si no está configurado manualmente.")
print("-" * 50)

# --- PASO 1: Definiciones de archivos y parámetros ---
print("--- PASO 1: Definiciones de archivos y parámetros ---")
original_properties_file_name = 'properties_2016.csv'
initial_cleaned_properties_file_name = 'properties_2016_initial_cleaned.csv'
original_train_file_name = 'train_2016_v2.csv'
base_train_merged_cleaned_file_name = 'train_final_cleaned_merged.csv'
output_train_with_zip_rank_file_name = 'train_ranked_by_zip_price.csv'
kaggle_dataset_name = 'zillow-prize-1'

dtype_properties = {
    'parcelid': np.int32, 'bathroomcnt': np.float32, 'bedroomcnt': np.float32,
    'calculatedfinishedsquarefeet': np.float32, 'yearbuilt': np.float32,
    'taxvaluedollarcnt': np.float32, 'taxamount': np.float32,
    'latitude': np.float32, 'longitude': np.float32,
    'regionidzip': np.float32, 'regionidcounty': np.float32
}
selected_property_features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                              'yearbuilt', 'taxvaluedollarcnt', 'taxamount', 'latitude', 'longitude',
                              'regionidzip', 'regionidcounty']
dtype_train = {'parcelid': np.int32}

print(f"Propiedades original: {original_properties_file_name}")
print(f"Propiedades con limpieza inicial: {initial_cleaned_properties_file_name}")
print(f"Transacciones original: {original_train_file_name}")
print(f"Base para este script (limpio y unido): {base_train_merged_cleaned_file_name}")
print(f"Archivo final con ranking de ZIP a generar: {output_train_with_zip_rank_file_name}")
print("-" * 50)

# --- PASO 2 & 3 & 4: Generar/Cargar base_train_merged_cleaned_file_name ---
success_generating_base_file = False
if not os.path.exists(base_train_merged_cleaned_file_name):
    print(f"--- PASO 2-4: '{base_train_merged_cleaned_file_name}' no existe. Intentando generarlo... ---")

    success_initial_cleaned_properties = False
    if not os.path.exists(initial_cleaned_properties_file_name):
        if not os.path.exists(original_properties_file_name):
            print(f"'{original_properties_file_name}' no encontrado. Descargando...")
            try:
                run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name}.zip --force")
                if os.path.exists(f'{original_properties_file_name}.zip'):
                    run_shell_command(f"unzip -o {original_properties_file_name}.zip")
                if not os.path.exists(original_properties_file_name): # Check if direct CSV downloaded
                     run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name} --force")
                if not os.path.exists(original_properties_file_name):
                    raise Exception(f"Fallo al descargar {original_properties_file_name}")
                print(f"'{original_properties_file_name}' descargado.")
            except Exception as e:
                print(f"Error descargando '{original_properties_file_name}': {e}")

        if os.path.exists(original_properties_file_name):
            try:
                print(f"Cargando '{original_properties_file_name}' para limpieza inicial...")
                df_props = pd.read_csv(original_properties_file_name, usecols=selected_property_features, dtype=dtype_properties)
                rows_before = len(df_props)
                df_props = df_props.dropna(subset=['yearbuilt'])
                df_props = df_props[df_props['bathroomcnt'] != 0]
                df_props = df_props[df_props['bedroomcnt'] != 0]
                df_props.to_csv(initial_cleaned_properties_file_name, index=False)
                print(f"'{initial_cleaned_properties_file_name}' generado (Filas: {len(df_props)} de {rows_before}).")
                success_initial_cleaned_properties = True
            except Exception as e:
                print(f"Error generando '{initial_cleaned_properties_file_name}': {e}")
        else:
            print(f"No se pudo generar '{initial_cleaned_properties_file_name}' porque '{original_properties_file_name}' no está disponible.")
    else:
        print(f"'{initial_cleaned_properties_file_name}' ya existe.")
        success_initial_cleaned_properties = True

    success_train_file_available = False
    if not os.path.exists(original_train_file_name):
        print(f"'{original_train_file_name}' no encontrado. Descargando...")
        try:
            run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_train_file_name} --force")
            if os.path.exists(original_train_file_name):
                success_train_file_available = True
            elif os.path.exists(f'{original_train_file_name}.zip'):
                run_shell_command(f"unzip -o {original_train_file_name}.zip")
                if os.path.exists(original_train_file_name):
                    success_train_file_available = True
            if not success_train_file_available:
                raise Exception(f"Fallo al descargar {original_train_file_name}")
            print(f"'{original_train_file_name}' descargado.")
        except Exception as e:
            print(f"Error descargando '{original_train_file_name}': {e}")
    else:
        print(f"'{original_train_file_name}' ya existe.")
        success_train_file_available = True

    if success_initial_cleaned_properties and success_train_file_available:
        try:
            print(f"Cargando '{initial_cleaned_properties_file_name}' y '{original_train_file_name}' para unir...")
            df_props_cleaned = pd.read_csv(initial_cleaned_properties_file_name, dtype=dtype_properties)
            if selected_property_features: # Asegurar que solo se usen las columnas seleccionadas si la lista no está vacía
                 df_props_cleaned = df_props_cleaned[[col for col in selected_property_features if col in df_props_cleaned.columns]]
            valid_parcelids = set(df_props_cleaned['parcelid'])

            df_train_orig = pd.read_csv(original_train_file_name, dtype=dtype_train, parse_dates=['transactiondate'])
            df_train_ids_filtered = df_train_orig[df_train_orig['parcelid'].isin(valid_parcelids)]

            print("Uniendo transacciones filtradas con datos de propiedades...")
            df_merged = pd.merge(df_train_ids_filtered, df_props_cleaned, on='parcelid', how='inner')
            print(f"Forma después de unir: {df_merged.shape}")

            if not df_merged.empty:
                print(f"Aplicando limpieza de NaNs a {df_merged.shape[0]} filas unidas...")
                columns_to_clean_nans = ['regionidzip', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet']
                # Verificar que las columnas existen
                actual_columns_to_clean = [col for col in columns_to_clean_nans if col in df_merged.columns]
                if not actual_columns_to_clean:
                     print("Advertencia: Ninguna de las columnas especificadas para limpieza de NaN se encontró. Omitiendo este paso.")
                     df_base_cleaned = df_merged.copy()
                else:
                    df_base_cleaned = df_merged.dropna(subset=actual_columns_to_clean)
                print(f"Forma después de limpieza de NaNs (columnas: {actual_columns_to_clean}): {df_base_cleaned.shape}")

                if not df_base_cleaned.empty:
                    df_base_cleaned.to_csv(base_train_merged_cleaned_file_name, index=False)
                    print(f"'{base_train_merged_cleaned_file_name}' generado y guardado.")
                    success_generating_base_file = True
                else:
                    print(f"'{base_train_merged_cleaned_file_name}' estaría vacío después de la limpieza de NaNs. No se guardó.")
            else:
                print("El DataFrame unido (df_merged) está vacío. No se puede proceder a la limpieza de NaNs.")
        except Exception as e:
            print(f"Error generando '{base_train_merged_cleaned_file_name}': {e}")
    else:
        print(f"No se pudo generar '{base_train_merged_cleaned_file_name}' por falta de archivos base.")

else:
    print(f"--- PASO 2-4: '{base_train_merged_cleaned_file_name}' ya existe. ---")
    success_generating_base_file = True
print("-" * 50)

# --- PASO 5: Cargar datos, crear columna 'price' y 'zip_price_rank' ---
df_ranked = None
median_price_per_zip_for_display = pd.DataFrame() # Para mostrar al final

if success_generating_base_file and os.path.exists(base_train_merged_cleaned_file_name):
    print(f"--- PASO 5: Creando ranking de precios por 'regionidzip' ---")
    try:
        df_final_cleaned = pd.read_csv(base_train_merged_cleaned_file_name)
        # Convertir columnas a tipos numéricos correctos si es necesario después de leer CSV
        for col, dtype_val in dtype_properties.items():
            if col in df_final_cleaned.columns:
                if pd.api.types.is_numeric_dtype(dtype_val): # float o int
                     df_final_cleaned[col] = pd.to_numeric(df_final_cleaned[col], errors='coerce')

        print(f"Cargado '{base_train_merged_cleaned_file_name}'. Forma: {df_final_cleaned.shape}")

        if df_final_cleaned.empty:
            print("El archivo base para ranking está vacío. No se puede continuar.")
        else:
            if 'taxvaluedollarcnt' in df_final_cleaned.columns and 'taxamount' in df_final_cleaned.columns:
                # Asegurar que no haya NaNs en estas dos columnas antes de la resta, o manejar el resultado NaN
                df_final_cleaned['price'] = df_final_cleaned['taxvaluedollarcnt'].fillna(0) - df_final_cleaned['taxamount'].fillna(0)
                # Alternativamente, si NaN en cualquiera debe resultar en NaN price:
                # df_final_cleaned['price'] = df_final_cleaned['taxvaluedollarcnt'] - df_final_cleaned['taxamount']
                print("Columna 'price' (taxvaluedollarcnt - taxamount) creada.")
            else:
                raise ValueError("Columnas 'taxvaluedollarcnt' o 'taxamount' no encontradas para crear 'price'.")

            print("Calculando mediana de 'price' por 'regionidzip'...")
            temp_df_for_median = df_final_cleaned.dropna(subset=['regionidzip', 'price'])
            if not temp_df_for_median.empty:
                median_price_per_zip = temp_df_for_median.groupby('regionidzip')['price'].median().reset_index()
                median_price_per_zip.rename(columns={'price': 'median_zip_price'}, inplace=True)
                print(f"Se calcularon medianas de precio para {len(median_price_per_zip)} regionidzip únicos.")
                median_price_per_zip_for_display = median_price_per_zip.copy() # Guardar para mostrar después
            else:
                print("No hay datos válidos para calcular medianas de precio por ZIP después de dropear NaNs en regionidzip/price.")
                median_price_per_zip = pd.DataFrame(columns=['regionidzip', 'median_zip_price']) # Dataframe vacío
                median_price_per_zip_for_display = median_price_per_zip.copy()

            if median_price_per_zip.empty and not temp_df_for_median.empty : # Chequeo extra por si groupby resultó vacío
                 print("Advertencia: El cálculo de mediana de precio por ZIP resultó en un DataFrame vacío aunque había datos de entrada. Verifique.")
                 # Para evitar error en .rank(), si está vacío, no creamos la columna o la llenamos con NaN
                 df_final_cleaned['zip_price_rank'] = np.nan
            elif not median_price_per_zip.empty:
                median_price_per_zip['zip_price_rank'] = median_price_per_zip['median_zip_price'].rank(method='min', ascending=True).astype(int)
                print("Ranking 'zip_price_rank' creado para cada regionidzip.")
                print("Uniendo 'zip_price_rank' al DataFrame principal...")
                df_ranked = pd.merge(df_final_cleaned, median_price_per_zip[['regionidzip', 'zip_price_rank']], on='regionidzip', how='left')
            else: # median_price_per_zip está vacío porque temp_df_for_median estaba vacío
                print("No se pudo crear 'zip_price_rank' porque no hay datos de 'regionidzip' o 'price' válidos.")
                df_ranked = df_final_cleaned.copy() # Copiar el dataframe sin el rank
                df_ranked['zip_price_rank'] = np.nan # Añadir la columna con NaNs

            print(f"Forma del DataFrame después de intentar unir el ranking: {df_ranked.shape if df_ranked is not None else 'N/A'}")

            if df_ranked is not None and 'zip_price_rank' in df_ranked.columns and df_ranked['zip_price_rank'].isnull().any():
                print(f"Advertencia: Se encontraron {df_ranked['zip_price_rank'].isnull().sum()} NaNs en 'zip_price_rank'.")

            if df_ranked is not None:
                df_ranked.to_csv(output_train_with_zip_rank_file_name, index=False)
                print(f"DataFrame con 'zip_price_rank' guardado en '{output_train_with_zip_rank_file_name}'.")

    except Exception as e:
        print(f"Error en PASO 5 (creación de ranking): {e}")
        df_ranked = None # Asegurar que df_ranked es None si hay error
else:
    print(f"No se puede proceder con PASO 5 porque '{base_train_merged_cleaned_file_name}' no está disponible o no se generó.")
print("-" * 50)

# --- PASO 6: Mostrar cabeza del DataFrame final con ranking ---
if df_ranked is not None and not df_ranked.empty:
    print(f"--- PASO 6: Primeras 5 filas del DataFrame final con 'zip_price_rank' ('{output_train_with_zip_rank_file_name}') ---")
    display_cols = ['parcelid', 'regionidzip', 'price', 'zip_price_rank']
    # Añadir logerror y transactiondate si existen
    if 'logerror' in df_ranked.columns: display_cols.append('logerror')
    if 'transactiondate' in df_ranked.columns: display_cols.append('transactiondate')
    # Filtrar display_cols para que solo contenga columnas que existen en df_ranked
    display_cols = [col for col in display_cols if col in df_ranked.columns]
    if display_cols:
        print(df_ranked[display_cols].head())
    else:
        print("No se pudieron seleccionar columnas para mostrar, mostrando todo el head:")
        print(df_ranked.head())

    if 'zip_price_rank' in df_ranked.columns:
        print("\nInformación sobre la nueva columna 'zip_price_rank':")
        print(df_ranked['zip_price_rank'].describe())

    if not median_price_per_zip_for_display.empty and 'zip_price_rank' in median_price_per_zip_for_display.columns:
        print("\nEjemplo de medianas de precio y ranks por ZIP (ordenado por rank):")
        print(median_price_per_zip_for_display.sort_values('zip_price_rank').head())
        print("...")
        print(median_price_per_zip_for_display.sort_values('zip_price_rank', ascending=False).head())
    elif not median_price_per_zip_for_display.empty:
        print("\nEjemplo de medianas de precio por ZIP (sin rankear o rank falló):")
        print(median_price_per_zip_for_display.head())


elif df_ranked is not None and df_ranked.empty:
     print(f"--- PASO 6: El DataFrame final ('{output_train_with_zip_rank_file_name}') está vacío. ---")
else:
    print(f"--- PASO 6: No se pudo generar el DataFrame final con ranking ('{output_train_with_zip_rank_file_name}'). ---")

# --- Resumen Final ---
if df_ranked is not None and not df_ranked.empty:
    print(f"\n✅ Proceso completado. El DataFrame con ranking de ZIP tiene {len(df_ranked)} filas.")
    print(f"Los datos finales se han guardado en '{output_train_with_zip_rank_file_name}'.")
elif df_ranked is not None and df_ranked.empty:
     print(f"\n⚠️ Proceso completado, pero el DataFrame final está vacío.")
else:
    print(f"\n❌ Proceso no completado. No se pudo generar el DataFrame con ranking de ZIP.")

import pandas as pd
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Función auxiliar para ejecutar comandos de shell
def run_shell_command(command):
    print(f"Ejecutando: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout: print(result.stdout)
        if result.stderr: print(f"Shell Error (stderr): {result.stderr}") # Prefixed to distinguish
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar comando: {command}\nError: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        raise

# --- PASO 0: CONFIGURACIÓN DE KAGGLE ---
print("Configurando Kaggle CLI...")
if os.path.exists('/content/kaggle.json'):
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    run_shell_command(f"cp /content/kaggle.json {os.path.expanduser('~/.kaggle/')}")
    run_shell_command(f"chmod 600 {os.path.expanduser('~/.kaggle/kaggle.json')}")
    print("Archivo kaggle.json configurado.")
else:
    print("Advertencia: /content/kaggle.json no encontrado. Descarga podría fallar.")
print("-" * 50)

# --- PASO 1: Definiciones de archivos y parámetros ---
print("--- PASO 1: Definiciones ---")
original_properties_file_name = 'properties_2016.csv'
initial_cleaned_properties_file_name = 'properties_2016_initial_cleaned.csv'
original_train_file_name = 'train_2016_v2.csv'
base_train_merged_cleaned_file_name = 'train_final_cleaned_merged.csv'
# This is the file we need as input for the KNN model
input_file_for_knn = 'train_ranked_by_zip_price.csv'

kaggle_dataset_name = 'zillow-prize-1'

dtype_properties = {
    'parcelid': np.int32, 'bathroomcnt': np.float32, 'bedroomcnt': np.float32,
    'calculatedfinishedsquarefeet': np.float32, 'yearbuilt': np.float32,
    'taxvaluedollarcnt': np.float32, 'taxamount': np.float32,
    'latitude': np.float32, 'longitude': np.float32,
    'regionidzip': np.float32, 'regionidcounty': np.float32
}
selected_property_features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                              'yearbuilt', 'taxvaluedollarcnt', 'taxamount', 'latitude', 'longitude',
                              'regionidzip', 'regionidcounty']
dtype_train = {'parcelid': np.int32}

print(f"Archivo de entrada para KNN (debe existir o se generará): {input_file_for_knn}")
print("-" * 50)

# --- PASO 2, 3, 4, 5a: Generar/Cargar input_file_for_knn ---
success_generating_input_knn_file = False
if not os.path.exists(input_file_for_knn):
    print(f"--- '{input_file_for_knn}' no existe. Intentando generarlo (esto puede tardar)... ---")

    # Sub-Step A: Generate initial_cleaned_properties_file_name
    success_initial_cleaned_properties = False
    if not os.path.exists(initial_cleaned_properties_file_name):
        if not os.path.exists(original_properties_file_name):
            print(f"Descargando '{original_properties_file_name}'...")
            try:
                run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name}.zip --force")
                if os.path.exists(f'{original_properties_file_name}.zip'): run_shell_command(f"unzip -o {original_properties_file_name}.zip")
                if not os.path.exists(original_properties_file_name): run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name} --force")
                if not os.path.exists(original_properties_file_name): raise Exception(f"Fallo descarga {original_properties_file_name}")
            except Exception as e: print(f"Error descargando '{original_properties_file_name}': {e}")

        if os.path.exists(original_properties_file_name):
            try:
                print(f"Limpiando '{original_properties_file_name}'...")
                df_props = pd.read_csv(original_properties_file_name, usecols=selected_property_features, dtype=dtype_properties)
                rows_before = len(df_props)
                df_props.dropna(subset=['yearbuilt'], inplace=True)
                df_props = df_props[df_props['bathroomcnt'] != 0]
                df_props = df_props[df_props['bedroomcnt'] != 0]
                df_props.to_csv(initial_cleaned_properties_file_name, index=False)
                print(f"'{initial_cleaned_properties_file_name}' generado (Filas: {len(df_props)} de {rows_before}).")
                success_initial_cleaned_properties = True
            except Exception as e: print(f"Error generando '{initial_cleaned_properties_file_name}': {e}")
        else: print(f"'{original_properties_file_name}' no disponible para generar '{initial_cleaned_properties_file_name}'.")
    else:
        print(f"'{initial_cleaned_properties_file_name}' ya existe.")
        success_initial_cleaned_properties = True

    # Sub-Step B: Ensure original_train_file_name is available
    success_train_file_available = False
    if not os.path.exists(original_train_file_name):
        print(f"Descargando '{original_train_file_name}'...")
        try:
            run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_train_file_name} --force")
            if os.path.exists(original_train_file_name): success_train_file_available = True
            elif os.path.exists(f'{original_train_file_name}.zip'):
                run_shell_command(f"unzip -o {original_train_file_name}.zip")
                if os.path.exists(original_train_file_name): success_train_file_available = True
            if not success_train_file_available: raise Exception(f"Fallo descarga {original_train_file_name}")
        except Exception as e: print(f"Error descargando '{original_train_file_name}': {e}")
    else:
        print(f"'{original_train_file_name}' ya existe.")
        success_train_file_available = True

    # Sub-Step C: Generate base_train_merged_cleaned_file_name
    success_generating_base_merged_file = False
    if not os.path.exists(base_train_merged_cleaned_file_name):
        if success_initial_cleaned_properties and success_train_file_available:
            try:
                print(f"Cargando y uniendo '{initial_cleaned_properties_file_name}' y '{original_train_file_name}'...")
                df_props_cleaned = pd.read_csv(initial_cleaned_properties_file_name, dtype=dtype_properties)
                df_props_cleaned = df_props_cleaned[[col for col in selected_property_features if col in df_props_cleaned.columns]]

                df_train_orig = pd.read_csv(original_train_file_name, dtype=dtype_train, parse_dates=['transactiondate'])
                valid_parcelids = set(df_props_cleaned['parcelid'])
                df_train_ids_filtered = df_train_orig[df_train_orig['parcelid'].isin(valid_parcelids)]

                df_merged = pd.merge(df_train_ids_filtered, df_props_cleaned, on='parcelid', how='inner')
                # print(f"Forma después de unir: {df_merged.shape}. Columnas: {df_merged.columns.tolist()}") # Descomentar para depurar

                if not df_merged.empty:
                    columns_to_clean_nans = ['regionidzip', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet']
                    actual_columns_to_clean = [col for col in columns_to_clean_nans if col in df_merged.columns]
                    df_base_cleaned = df_merged.dropna(subset=actual_columns_to_clean) if actual_columns_to_clean else df_merged.copy()

                    if not df_base_cleaned.empty:
                        df_base_cleaned.to_csv(base_train_merged_cleaned_file_name, index=False)
                        print(f"'{base_train_merged_cleaned_file_name}' generado.")
                        success_generating_base_merged_file = True
                    else: print(f"'{base_train_merged_cleaned_file_name}' estaría vacío. No se guardó.")
                else: print("DataFrame unido vacío.")
            except Exception as e: print(f"Error generando '{base_train_merged_cleaned_file_name}': {e}")
        else: print(f"'{base_train_merged_cleaned_file_name}' no generado por falta de archivos base.")
    else:
        print(f"'{base_train_merged_cleaned_file_name}' ya existe.")
        success_generating_base_merged_file = True

    # Sub-Step D: Add 'price' and 'zip_price_rank' to create input_file_for_knn
    if success_generating_base_merged_file and os.path.exists(base_train_merged_cleaned_file_name):
        try:
            print(f"Creando '{input_file_for_knn}' a partir de '{base_train_merged_cleaned_file_name}'...")
            df_loaded_base = pd.read_csv(base_train_merged_cleaned_file_name)
            if not df_loaded_base.empty:
                df_to_rank = df_loaded_base.copy()
                if 'taxvaluedollarcnt' in df_to_rank.columns and 'taxamount' in df_to_rank.columns:
                    df_to_rank['price'] = df_to_rank['taxvaluedollarcnt'].fillna(0) - df_to_rank['taxamount'].fillna(0)
                else: raise ValueError("Columnas para 'price' no encontradas.")

                temp_df_for_median = df_to_rank.dropna(subset=['regionidzip', 'price'])
                if not temp_df_for_median.empty:
                    median_price_per_zip = temp_df_for_median.groupby('regionidzip')['price'].median().reset_index()
                    median_price_per_zip.rename(columns={'price': 'median_zip_price'}, inplace=True)
                    if not median_price_per_zip.empty:
                        median_price_per_zip['zip_price_rank'] = median_price_per_zip['median_zip_price'].rank(method='min', ascending=True).astype(int)
                        df_ranked = pd.merge(df_to_rank, median_price_per_zip[['regionidzip', 'zip_price_rank']], on='regionidzip', how='left')
                    else:
                        df_ranked = df_to_rank.copy()
                        df_ranked['zip_price_rank'] = np.nan
                else:
                    df_ranked = df_to_rank.copy()
                    df_ranked['zip_price_rank'] = np.nan

                df_ranked.to_csv(input_file_for_knn, index=False)
                print(f"'{input_file_for_knn}' generado y guardado.")
                success_generating_input_knn_file = True
            else: print(f"'{base_train_merged_cleaned_file_name}' está vacío.")
        except Exception as e: print(f"Error generando '{input_file_for_knn}': {e}")
    else:
        print(f"No se pudo generar '{input_file_for_knn}' porque '{base_train_merged_cleaned_file_name}' no está disponible.")
else:
    print(f"--- '{input_file_for_knn}' ya existe. ---")
    success_generating_input_knn_file = True
print("-" * 50)

# --- PASO 5b: Cargar datos para el modelo KNN ---
df_model_input = None
data_for_knn_analysis = pd.DataFrame() # Para usar en el bloque final

if success_generating_input_knn_file and os.path.exists(input_file_for_knn):
    print(f"--- PASO 5b: Cargando datos para el modelo KNN desde '{input_file_for_knn}' ---")
    try:
        df_model_input = pd.read_csv(input_file_for_knn)
        print(f"Cargado '{input_file_for_knn}'. Forma: {df_model_input.shape}")
        # print("Columnas disponibles:", df_model_input.columns.tolist()) # Descomentar para depurar
    except Exception as e:
        print(f"Error cargando '{input_file_for_knn}': {e}")
        df_model_input = None
else:
    print(f"No se puede proceder con el modelo KNN: '{input_file_for_knn}' no disponible.")
print("-" * 50)

# --- PASO 6: Preparación de datos para KNN ---
if df_model_input is not None and not df_model_input.empty:
    print(f"--- PASO 6: Preparando datos para KNN ---")

    # ***** CAMBIO PRINCIPAL AQUÍ *****
    target_col = 'price'
    feature_cols = ['zip_price_rank', 'yearbuilt', 'calculatedfinishedsquarefeet', 'bathroomcnt', 'bedroomcnt']
    # 'price' se ha movido a target_col y eliminado de feature_cols
    # ***** FIN CAMBIO PRINCIPAL *****

    # Verificar que todas las columnas necesarias existen
    all_needed_cols = feature_cols + [target_col]
    missing_cols = [col for col in all_needed_cols if col not in df_model_input.columns]

    if missing_cols:
        print(f"Error: Faltan las siguientes columnas necesarias para el modelo KNN: {missing_cols}")
        df_model_input = None
    else:
        print(f"Features seleccionadas: {feature_cols}")
        print(f"Target seleccionado: {target_col}")

        X = df_model_input[feature_cols]
        y = df_model_input[target_col]

        # Manejar NaNs restantes en X o y (importante para KNN)
        # Crear un DataFrame temporal para eliminar NaNs de forma alineada
        temp_df_for_nan_cleaning = X.copy()
        temp_df_for_nan_cleaning[target_col] = y

        rows_before_nan_drop = len(temp_df_for_nan_cleaning)
        temp_df_for_nan_cleaning.dropna(inplace=True)
        rows_after_nan_drop = len(temp_df_for_nan_cleaning)
        print(f"Filas eliminadas por NaNs en features/target seleccionados: {rows_before_nan_drop - rows_after_nan_drop}")

        data_for_knn_analysis = temp_df_for_nan_cleaning # Guardar para el bloque final de resumen

        if data_for_knn_analysis.empty:
            print("No quedan datos después de eliminar NaNs. No se puede continuar con el modelo KNN.")
            df_model_input = None
        else:
            X = data_for_knn_analysis[feature_cols]
            y = data_for_knn_analysis[target_col]

            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Datos divididos: {len(X_train)} entrenamiento ({len(X_train)/len(X)*100:.2f}%), {len(X_test)} prueba ({len(X_test)/len(X)*100:.2f}%).")

            # Escalar Features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            print("Features escaladas.")

            # --- PASO 7: Análisis de `k` óptimo para KNN ---
            print("\n--- PASO 7: Analizando k óptimo para KNN Regressor (prediciendo 'price') ---")
            k_range = range(1, 31) # Probar k de 1 a 30 (ajustado para no ser demasiado largo)
            mse_values = []

            print("Calculando MSE para diferentes valores de k...")
            for k_val in k_range:
                knn_model = KNeighborsRegressor(n_neighbors=k_val)
                knn_model.fit(X_train_scaled, y_train)
                y_pred = knn_model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                mse_values.append(mse)
                # print(f"k = {k_val}, MSE = {mse:.2f}") # Descomentar para ver MSE por k

            # Graficar MSE vs. k
            plt.figure(figsize=(12, 6))
            plt.plot(k_range, mse_values, marker='o', linestyle='-')
            plt.title(f'MSE vs. Número de Vecinos (k) para KNN Regressor (Prediciendo {target_col})')
            plt.xlabel('Número de Vecinos (k)')
            plt.ylabel('Mean Squared Error (MSE)')
            plt.xticks(list(k_range)) # Muestra todos los k en el eje x si el rango es pequeño
            plt.grid(True)
            plt.show()

            if mse_values: # Asegurar que mse_values no está vacío
                optimal_k_mse = min(mse_values)
                optimal_k_value = k_range[mse_values.index(optimal_k_mse)]
                print(f"\nEl MSE mínimo es {optimal_k_mse:.2f} para k = {optimal_k_value}.")
            else:
                print("\nNo se pudieron calcular los valores de MSE.")
            print("Busca en el gráfico un 'codo' o el punto donde el MSE deja de disminuir significativamente.")

else:
    if df_model_input is not None and df_model_input.empty :
         print("El DataFrame de entrada para el modelo KNN está vacío.")
    else: # df_model_input es None
         print("No se pudo cargar o generar el DataFrame de entrada para el modelo KNN.")

print("-" * 50)
# --- Resumen Final ---
if df_model_input is not None and not data_for_knn_analysis.empty:
    print("\n✅ Proceso de análisis de k para KNN completado.")
    print(f"Se analizó k para predecir '{target_col}' usando las features: {feature_cols}.")
else:
    print("\n❌ Proceso de análisis de k para KNN no pudo completarse debido a problemas con los datos.")

import pandas as pd
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error # NUEVAS IMPORTACIONES

# Función auxiliar para ejecutar comandos de shell
def run_shell_command(command):
    print(f"Ejecutando: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout: print(result.stdout)
        if result.stderr: print(f"Shell Error (stderr): {result.stderr}") # Prefixed to distinguish
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar comando: {command}\nError: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        raise

# --- PASO 0: CONFIGURACIÓN DE KAGGLE ---
print("Configurando Kaggle CLI...")
if os.path.exists('/content/kaggle.json'):
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    run_shell_command(f"cp /content/kaggle.json {os.path.expanduser('~/.kaggle/')}")
    run_shell_command(f"chmod 600 {os.path.expanduser('~/.kaggle/kaggle.json')}")
    print("Archivo kaggle.json configurado.")
else:
    print("Advertencia: /content/kaggle.json no encontrado. Descarga podría fallar.")
print("-" * 50)

# --- PASO 1: Definiciones de archivos y parámetros ---
print("--- PASO 1: Definiciones ---")
original_properties_file_name = 'properties_2016.csv'
initial_cleaned_properties_file_name = 'properties_2016_initial_cleaned.csv'
original_train_file_name = 'train_2016_v2.csv'
base_train_merged_cleaned_file_name = 'train_final_cleaned_merged.csv'
# This is the file we need as input for the KNN model
input_file_for_knn = 'train_ranked_by_zip_price.csv'

kaggle_dataset_name = 'zillow-prize-1'

dtype_properties = {
    'parcelid': np.int32, 'bathroomcnt': np.float32, 'bedroomcnt': np.float32,
    'calculatedfinishedsquarefeet': np.float32, 'yearbuilt': np.float32,
    'taxvaluedollarcnt': np.float32, 'taxamount': np.float32,
    'latitude': np.float32, 'longitude': np.float32,
    'regionidzip': np.float32, 'regionidcounty': np.float32
}
selected_property_features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                              'yearbuilt', 'taxvaluedollarcnt', 'taxamount', 'latitude', 'longitude',
                              'regionidzip', 'regionidcounty']
dtype_train = {'parcelid': np.int32}

print(f"Archivo de entrada para KNN (debe existir o se generará): {input_file_for_knn}")
print("-" * 50)

# --- PASO 2, 3, 4, 5a: Generar/Cargar input_file_for_knn ---
success_generating_input_knn_file = False
if not os.path.exists(input_file_for_knn):
    print(f"--- '{input_file_for_knn}' no existe. Intentando generarlo (esto puede tardar)... ---")

    # Sub-Step A: Generate initial_cleaned_properties_file_name
    success_initial_cleaned_properties = False
    if not os.path.exists(initial_cleaned_properties_file_name):
        if not os.path.exists(original_properties_file_name):
            print(f"Descargando '{original_properties_file_name}'...")
            try:
                run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name}.zip --force")
                if os.path.exists(f'{original_properties_file_name}.zip'): run_shell_command(f"unzip -o {original_properties_file_name}.zip")
                if not os.path.exists(original_properties_file_name): run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name} --force")
                if not os.path.exists(original_properties_file_name): raise Exception(f"Fallo descarga {original_properties_file_name}")
            except Exception as e: print(f"Error descargando '{original_properties_file_name}': {e}")

        if os.path.exists(original_properties_file_name):
            try:
                print(f"Limpiando '{original_properties_file_name}'...")
                df_props = pd.read_csv(original_properties_file_name, usecols=selected_property_features, dtype=dtype_properties)
                rows_before = len(df_props)
                df_props.dropna(subset=['yearbuilt'], inplace=True)
                df_props = df_props[df_props['bathroomcnt'] != 0]
                df_props = df_props[df_props['bedroomcnt'] != 0]
                df_props.to_csv(initial_cleaned_properties_file_name, index=False)
                print(f"'{initial_cleaned_properties_file_name}' generado (Filas: {len(df_props)} de {rows_before}).")
                success_initial_cleaned_properties = True
            except Exception as e: print(f"Error generando '{initial_cleaned_properties_file_name}': {e}")
        else: print(f"'{original_properties_file_name}' no disponible para generar '{initial_cleaned_properties_file_name}'.")
    else:
        print(f"'{initial_cleaned_properties_file_name}' ya existe.")
        success_initial_cleaned_properties = True

    # Sub-Step B: Ensure original_train_file_name is available
    success_train_file_available = False
    if not os.path.exists(original_train_file_name):
        print(f"Descargando '{original_train_file_name}'...")
        try:
            run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_train_file_name} --force")
            if os.path.exists(original_train_file_name): success_train_file_available = True
            elif os.path.exists(f'{original_train_file_name}.zip'):
                run_shell_command(f"unzip -o {original_train_file_name}.zip")
                if os.path.exists(original_train_file_name): success_train_file_available = True
            if not success_train_file_available: raise Exception(f"Fallo descarga {original_train_file_name}")
        except Exception as e: print(f"Error descargando '{original_train_file_name}': {e}")
    else:
        print(f"'{original_train_file_name}' ya existe.")
        success_train_file_available = True

    # Sub-Step C: Generate base_train_merged_cleaned_file_name
    success_generating_base_merged_file = False
    if not os.path.exists(base_train_merged_cleaned_file_name):
        if success_initial_cleaned_properties and success_train_file_available:
            try:
                print(f"Cargando y uniendo '{initial_cleaned_properties_file_name}' y '{original_train_file_name}'...")
                df_props_cleaned = pd.read_csv(initial_cleaned_properties_file_name, dtype=dtype_properties)
                df_props_cleaned = df_props_cleaned[[col for col in selected_property_features if col in df_props_cleaned.columns]]

                df_train_orig = pd.read_csv(original_train_file_name, dtype=dtype_train, parse_dates=['transactiondate'])
                valid_parcelids = set(df_props_cleaned['parcelid'])
                df_train_ids_filtered = df_train_orig[df_train_orig['parcelid'].isin(valid_parcelids)]

                df_merged = pd.merge(df_train_ids_filtered, df_props_cleaned, on='parcelid', how='inner')
                # print(f"Forma después de unir: {df_merged.shape}. Columnas: {df_merged.columns.tolist()}") # Descomentar para depurar

                if not df_merged.empty:
                    columns_to_clean_nans = ['regionidzip', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet']
                    actual_columns_to_clean = [col for col in columns_to_clean_nans if col in df_merged.columns]
                    df_base_cleaned = df_merged.dropna(subset=actual_columns_to_clean) if actual_columns_to_clean else df_merged.copy()

                    if not df_base_cleaned.empty:
                        df_base_cleaned.to_csv(base_train_merged_cleaned_file_name, index=False)
                        print(f"'{base_train_merged_cleaned_file_name}' generado.")
                        success_generating_base_merged_file = True
                    else: print(f"'{base_train_merged_cleaned_file_name}' estaría vacío. No se guardó.")
                else: print("DataFrame unido vacío.")
            except Exception as e: print(f"Error generando '{base_train_merged_cleaned_file_name}': {e}")
        else: print(f"'{base_train_merged_cleaned_file_name}' no generado por falta de archivos base.")
    else:
        print(f"'{base_train_merged_cleaned_file_name}' ya existe.")
        success_generating_base_merged_file = True

    # Sub-Step D: Add 'price' and 'zip_price_rank' to create input_file_for_knn
    if success_generating_base_merged_file and os.path.exists(base_train_merged_cleaned_file_name):
        try:
            print(f"Creando '{input_file_for_knn}' a partir de '{base_train_merged_cleaned_file_name}'...")
            df_loaded_base = pd.read_csv(base_train_merged_cleaned_file_name)
            if not df_loaded_base.empty:
                df_to_rank = df_loaded_base.copy()
                if 'taxvaluedollarcnt' in df_to_rank.columns and 'taxamount' in df_to_rank.columns:
                    df_to_rank['price'] = df_to_rank['taxvaluedollarcnt'].fillna(0) - df_to_rank['taxamount'].fillna(0)
                else: raise ValueError("Columnas para 'price' no encontradas.")

                temp_df_for_median = df_to_rank.dropna(subset=['regionidzip', 'price'])
                if not temp_df_for_median.empty:
                    median_price_per_zip = temp_df_for_median.groupby('regionidzip')['price'].median().reset_index()
                    median_price_per_zip.rename(columns={'price': 'median_zip_price'}, inplace=True)
                    if not median_price_per_zip.empty:
                        median_price_per_zip['zip_price_rank'] = median_price_per_zip['median_zip_price'].rank(method='min', ascending=True).astype(int)
                        df_ranked = pd.merge(df_to_rank, median_price_per_zip[['regionidzip', 'zip_price_rank']], on='regionidzip', how='left')
                    else:
                        df_ranked = df_to_rank.copy()
                        df_ranked['zip_price_rank'] = np.nan
                else:
                    df_ranked = df_to_rank.copy()
                    df_ranked['zip_price_rank'] = np.nan

                df_ranked.to_csv(input_file_for_knn, index=False)
                print(f"'{input_file_for_knn}' generado y guardado.")
                success_generating_input_knn_file = True
            else: print(f"'{base_train_merged_cleaned_file_name}' está vacío.")
        except Exception as e: print(f"Error generando '{input_file_for_knn}': {e}")
    else:
        print(f"No se pudo generar '{input_file_for_knn}' porque '{base_train_merged_cleaned_file_name}' no está disponible.")
else:
    print(f"--- '{input_file_for_knn}' ya existe. ---")
    success_generating_input_knn_file = True
print("-" * 50)

# --- PASO 5b: Cargar datos para el modelo KNN ---
df_model_input = None
data_for_knn_analysis = pd.DataFrame() # Para usar en el bloque final

if success_generating_input_knn_file and os.path.exists(input_file_for_knn):
    print(f"--- PASO 5b: Cargando datos para el modelo KNN desde '{input_file_for_knn}' ---")
    try:
        df_model_input = pd.read_csv(input_file_for_knn)
        print(f"Cargado '{input_file_for_knn}'. Forma: {df_model_input.shape}")
        # print("Columnas disponibles:", df_model_input.columns.tolist()) # Descomentar para depurar
    except Exception as e:
        print(f"Error cargando '{input_file_for_knn}': {e}")
        df_model_input = None
else:
    print(f"No se puede proceder con el modelo KNN: '{input_file_for_knn}' no disponible.")
print("-" * 50)

# --- PASO 6: Preparación de datos para KNN ---
if df_model_input is not None and not df_model_input.empty:
    print(f"--- PASO 6: Preparando datos para KNN ---")

    target_col = 'price'
    feature_cols = ['zip_price_rank', 'yearbuilt', 'calculatedfinishedsquarefeet', 'bathroomcnt', 'bedroomcnt']

    all_needed_cols = feature_cols + [target_col]
    missing_cols = [col for col in all_needed_cols if col not in df_model_input.columns]

    if missing_cols:
        print(f"Error: Faltan las siguientes columnas necesarias para el modelo KNN: {missing_cols}")
        df_model_input = None
    else:
        print(f"Features seleccionadas: {feature_cols}")
        print(f"Target seleccionado: {target_col}")

        X = df_model_input[feature_cols]
        y = df_model_input[target_col]

        temp_df_for_nan_cleaning = X.copy()
        temp_df_for_nan_cleaning[target_col] = y

        rows_before_nan_drop = len(temp_df_for_nan_cleaning)
        temp_df_for_nan_cleaning.dropna(inplace=True)
        rows_after_nan_drop = len(temp_df_for_nan_cleaning)
        print(f"Filas eliminadas por NaNs en features/target seleccionados: {rows_before_nan_drop - rows_after_nan_drop}")

        data_for_knn_analysis = temp_df_for_nan_cleaning

        if data_for_knn_analysis.empty:
            print("No quedan datos después de eliminar NaNs. No se puede continuar con el modelo KNN.")
            df_model_input = None
        else:
            X = data_for_knn_analysis[feature_cols]
            y = data_for_knn_analysis[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Datos divididos: {len(X_train)} entrenamiento ({len(X_train)/len(X)*100:.2f}%), {len(X_test)} prueba ({len(X_test)/len(X)*100:.2f}%).")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            print("Features escaladas.")

            # --- PASO 7: Análisis de `k` óptimo para KNN ---
            print("\n--- PASO 7: Analizando k óptimo para KNN Regressor (prediciendo 'price') ---")
            k_range = range(1, 31)
            mse_values = []

            print("Calculando MSE para diferentes valores de k...")
            for k_val in k_range:
                knn_model = KNeighborsRegressor(n_neighbors=k_val)
                knn_model.fit(X_train_scaled, y_train)
                y_pred = knn_model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                mse_values.append(mse)
                # print(f"k = {k_val}, MSE = {mse:.2f}")

            plt.figure(figsize=(12, 6))
            plt.plot(k_range, mse_values, marker='o', linestyle='-')
            plt.title(f'MSE vs. Número de Vecinos (k) para KNN Regressor (Prediciendo {target_col})')
            plt.xlabel('Número de Vecinos (k)')
            plt.ylabel('Mean Squared Error (MSE)')
            plt.xticks(list(k_range))
            plt.grid(True)
            plt.show()

            if mse_values:
                optimal_k_mse = min(mse_values)
                optimal_k_value = k_range[mse_values.index(optimal_k_mse)]
                print(f"\nEl MSE mínimo es {optimal_k_mse:.2f} para k = {optimal_k_value}.")
            else:
                print("\nNo se pudieron calcular los valores de MSE.")
            print("Busca en el gráfico un 'codo' o el punto donde el MSE deja de disminuir significativamente.")

            # --- PASO 8 (NUEVO): Evaluación detallada con k=8 (basado en tu elección de codo) ---
            print("\n--- PASO 8: Evaluación detallada con k=8 ---")
            k_elegido = 22 # El valor de k que identificaste como el "codo"

            # Asegurarse de que k_elegido esté dentro del rango de k_values si queremos tomar el mse de la lista
            # o simplemente re-entrenar y predecir, que es más limpio.

            print(f"Evaluando el modelo con k={k_elegido}...")

            # 1. Entrenar el modelo KNN con k_elegido
            knn_model_elegido = KNeighborsRegressor(n_neighbors=k_elegido)
            knn_model_elegido.fit(X_train_scaled, y_train)

            # 2. Realizar predicciones en el conjunto de prueba
            y_pred_elegido = knn_model_elegido.predict(X_test_scaled)

            # 3. Calcular MSE y RMSE para k_elegido
            mse_elegido = mean_squared_error(y_test, y_pred_elegido)
            rmse_elegido = np.sqrt(mse_elegido)
            print(f"  Mean Squared Error (MSE) para k={k_elegido}: {mse_elegido:.2f}")
            print(f"  Root Mean Squared Error (RMSE) para k={k_elegido}: {rmse_elegido:.2f} (error promedio en unidades de '{target_col}')")

            # 4. Calcular R-cuadrado (R²)
            r2_elegido = r2_score(y_test, y_pred_elegido)
            print(f"  Coeficiente de Determinación (R²) para k={k_elegido}: {r2_elegido:.4f}")
            print(f"  Esto significa que el modelo con k={k_elegido} explica aproximadamente el {r2_elegido*100:.2f}% de la variabilidad en '{target_col}'.")

            # 5. Calcular MAPE
            # ¡Importante! MAPE es sensible a valores cero o negativos en y_test.
            if np.all(y_test > 0): # Una verificación simple para valores positivos
                mape_elegido = mean_absolute_percentage_error(y_test, y_pred_elegido)
                print(f"  Error Porcentual Absoluto Medio (MAPE) para k={k_elegido}: {mape_elegido:.4f} ({mape_elegido*100:.2f}%)")
                print(f"  Esto significa que, en promedio, las predicciones del modelo con k={k_elegido} se desvían en un {mape_elegido*100:.2f}% del valor real de '{target_col}'.")
                if mape_elegido < 1: # Solo tiene sentido si MAPE es < 1 (o < 100%)
                     print(f"  Podrías interpretar una 'efectividad porcentual' (basada en MAPE) como {(1-mape_elegido)*100:.2f}%.")
            else:
                print(f"  MAPE no se calculó o podría no ser fiable para k={k_elegido} porque '{target_col}' (y_test) contiene valores cero o negativos.")
                print(f"  Considera R² y RMSE como las principales métricas de rendimiento en este caso.")
            # --- FIN PASO 8 ---

else:
    if df_model_input is not None and df_model_input.empty :
         print("El DataFrame de entrada para el modelo KNN está vacío.")
    else: # df_model_input es None
         print("No se pudo cargar o generar el DataFrame de entrada para el modelo KNN.")

print("-" * 50)
# --- Resumen Final ---
if df_model_input is not None and not data_for_knn_analysis.empty:
    print("\n✅ Proceso de análisis de k para KNN completado.")
    if 'k_elegido' in locals() or 'optimal_k_value' in locals(): # Verifica si se realizó la evaluación
        k_final_eval = k_elegido if 'k_elegido' in locals() else optimal_k_value
        print(f"Se evaluaron métricas adicionales (incluyendo R² y MAPE si aplica) para k={k_final_eval}.")
    print(f"Se analizó k para predecir '{target_col}' usando las features: {feature_cols}.")
else:
    print("\n❌ Proceso de análisis de k para KNN no pudo completarse debido a problemas con los datos.")
import pandas as pd
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Función auxiliar para ejecutar comandos de shell
def run_shell_command(command):
    print(f"Ejecutando: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout: print(result.stdout)
        if result.stderr: print(f"Shell Error (stderr): {result.stderr}") # Prefixed to distinguish
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar comando: {command}\nError: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        raise

# --- PASO 0: CONFIGURACIÓN DE KAGGLE ---
print("Configurando Kaggle CLI...")
if os.path.exists('/content/kaggle.json'):
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    run_shell_command(f"cp /content/kaggle.json {os.path.expanduser('~/.kaggle/')}")
    run_shell_command(f"chmod 600 {os.path.expanduser('~/.kaggle/kaggle.json')}")
    print("Archivo kaggle.json configurado.")
else:
    print("Advertencia: /content/kaggle.json no encontrado. Descarga podría fallar.")
print("-" * 50)

# --- PASO 1: Definiciones de archivos y parámetros ---
print("--- PASO 1: Definiciones ---")
original_properties_file_name = 'properties_2016.csv'
initial_cleaned_properties_file_name = 'properties_2016_initial_cleaned.csv'
original_train_file_name = 'train_2016_v2.csv'
base_train_merged_cleaned_file_name = 'train_final_cleaned_merged.csv'
# This is the file we need as input for the KNN model
input_file_for_knn = 'train_ranked_by_zip_price.csv'

kaggle_dataset_name = 'zillow-prize-1'

dtype_properties = {
    'parcelid': np.int32, 'bathroomcnt': np.float32, 'bedroomcnt': np.float32,
    'calculatedfinishedsquarefeet': np.float32, 'yearbuilt': np.float32,
    'taxvaluedollarcnt': np.float32, 'taxamount': np.float32,
    'latitude': np.float32, 'longitude': np.float32,
    'regionidzip': np.float32, 'regionidcounty': np.float32
}
selected_property_features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                              'yearbuilt', 'taxvaluedollarcnt', 'taxamount', 'latitude', 'longitude',
                              'regionidzip', 'regionidcounty']
dtype_train = {'parcelid': np.int32}

print(f"Archivo de entrada para KNN (debe existir o se generará): {input_file_for_knn}")
print("-" * 50)

# --- PASO 2, 3, 4, 5a: Generar/Cargar input_file_for_knn ---
success_generating_input_knn_file = False
if not os.path.exists(input_file_for_knn):
    print(f"--- '{input_file_for_knn}' no existe. Intentando generarlo (esto puede tardar)... ---")

    # Sub-Step A: Generate initial_cleaned_properties_file_name
    success_initial_cleaned_properties = False
    if not os.path.exists(initial_cleaned_properties_file_name):
        if not os.path.exists(original_properties_file_name):
            print(f"Descargando '{original_properties_file_name}'...")
            try:
                run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name}.zip --force")
                if os.path.exists(f'{original_properties_file_name}.zip'): run_shell_command(f"unzip -o {original_properties_file_name}.zip")
                if not os.path.exists(original_properties_file_name): run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name} --force")
                if not os.path.exists(original_properties_file_name): raise Exception(f"Fallo descarga {original_properties_file_name}")
            except Exception as e: print(f"Error descargando '{original_properties_file_name}': {e}")

        if os.path.exists(original_properties_file_name):
            try:
                print(f"Limpiando '{original_properties_file_name}'...")
                df_props = pd.read_csv(original_properties_file_name, usecols=selected_property_features, dtype=dtype_properties)
                rows_before = len(df_props)
                df_props.dropna(subset=['yearbuilt'], inplace=True) # Asegura que 'yearbuilt' no tenga NaNs antes de usarlo
                df_props = df_props[df_props['bathroomcnt'] != 0]
                df_props = df_props[df_props['bedroomcnt'] != 0]
                df_props.to_csv(initial_cleaned_properties_file_name, index=False)
                print(f"'{initial_cleaned_properties_file_name}' generado (Filas: {len(df_props)} de {rows_before}).")
                success_initial_cleaned_properties = True
            except Exception as e: print(f"Error generando '{initial_cleaned_properties_file_name}': {e}")
        else: print(f"'{original_properties_file_name}' no disponible para generar '{initial_cleaned_properties_file_name}'.")
    else:
        print(f"'{initial_cleaned_properties_file_name}' ya existe.")
        success_initial_cleaned_properties = True

    # Sub-Step B: Ensure original_train_file_name is available
    success_train_file_available = False
    if not os.path.exists(original_train_file_name):
        print(f"Descargando '{original_train_file_name}'...")
        try:
            run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_train_file_name} --force")
            if os.path.exists(original_train_file_name): success_train_file_available = True
            elif os.path.exists(f'{original_train_file_name}.zip'):
                run_shell_command(f"unzip -o {original_train_file_name}.zip")
                if os.path.exists(original_train_file_name): success_train_file_available = True
            if not success_train_file_available: raise Exception(f"Fallo descarga {original_train_file_name}")
        except Exception as e: print(f"Error descargando '{original_train_file_name}': {e}")
    else:
        print(f"'{original_train_file_name}' ya existe.")
        success_train_file_available = True

    # Sub-Step C: Generate base_train_merged_cleaned_file_name
    success_generating_base_merged_file = False
    if not os.path.exists(base_train_merged_cleaned_file_name):
        if success_initial_cleaned_properties and success_train_file_available:
            try:
                print(f"Cargando y uniendo '{initial_cleaned_properties_file_name}' y '{original_train_file_name}'...")
                df_props_cleaned = pd.read_csv(initial_cleaned_properties_file_name, dtype=dtype_properties)
                df_props_cleaned = df_props_cleaned[[col for col in selected_property_features if col in df_props_cleaned.columns]]

                df_train_orig = pd.read_csv(original_train_file_name, dtype=dtype_train, parse_dates=['transactiondate'])
                valid_parcelids = set(df_props_cleaned['parcelid'])
                df_train_ids_filtered = df_train_orig[df_train_orig['parcelid'].isin(valid_parcelids)]

                df_merged = pd.merge(df_train_ids_filtered, df_props_cleaned, on='parcelid', how='inner')

                if not df_merged.empty:
                    columns_to_clean_nans = ['regionidzip', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet'] # 'yearbuilt' ya no tiene NaNs
                    actual_columns_to_clean = [col for col in columns_to_clean_nans if col in df_merged.columns]
                    df_base_cleaned = df_merged.dropna(subset=actual_columns_to_clean) if actual_columns_to_clean else df_merged.copy()

                    if not df_base_cleaned.empty:
                        df_base_cleaned.to_csv(base_train_merged_cleaned_file_name, index=False)
                        print(f"'{base_train_merged_cleaned_file_name}' generado.")
                        success_generating_base_merged_file = True
                    else: print(f"'{base_train_merged_cleaned_file_name}' estaría vacío. No se guardó.")
                else: print("DataFrame unido vacío.")
            except Exception as e: print(f"Error generando '{base_train_merged_cleaned_file_name}': {e}")
        else: print(f"'{base_train_merged_cleaned_file_name}' no generado por falta de archivos base.")
    else:
        print(f"'{base_train_merged_cleaned_file_name}' ya existe.")
        success_generating_base_merged_file = True

    # Sub-Step D: Add 'price' and 'zip_price_rank' to create input_file_for_knn
    if success_generating_base_merged_file and os.path.exists(base_train_merged_cleaned_file_name):
        try:
            print(f"Creando '{input_file_for_knn}' a partir de '{base_train_merged_cleaned_file_name}'...")
            df_loaded_base = pd.read_csv(base_train_merged_cleaned_file_name)
            if not df_loaded_base.empty:
                df_to_rank = df_loaded_base.copy()
                if 'taxvaluedollarcnt' in df_to_rank.columns and 'taxamount' in df_to_rank.columns:
                    df_to_rank['price'] = df_to_rank['taxvaluedollarcnt'].fillna(0) - df_to_rank['taxamount'].fillna(0)
                else: raise ValueError("Columnas para 'price' no encontradas.")

                temp_df_for_median = df_to_rank.dropna(subset=['regionidzip', 'price'])
                if not temp_df_for_median.empty:
                    median_price_per_zip = temp_df_for_median.groupby('regionidzip')['price'].median().reset_index()
                    median_price_per_zip.rename(columns={'price': 'median_zip_price'}, inplace=True)
                    if not median_price_per_zip.empty:
                        median_price_per_zip['zip_price_rank'] = median_price_per_zip['median_zip_price'].rank(method='min', ascending=True).astype(int)
                        df_ranked = pd.merge(df_to_rank, median_price_per_zip[['regionidzip', 'zip_price_rank']], on='regionidzip', how='left')
                    else:
                        df_ranked = df_to_rank.copy()
                        df_ranked['zip_price_rank'] = np.nan
                else:
                    df_ranked = df_to_rank.copy()
                    df_ranked['zip_price_rank'] = np.nan

                df_ranked.to_csv(input_file_for_knn, index=False)
                print(f"'{input_file_for_knn}' generado y guardado.")
                success_generating_input_knn_file = True
            else: print(f"'{base_train_merged_cleaned_file_name}' está vacío.")
        except Exception as e: print(f"Error generando '{input_file_for_knn}': {e}")
    else:
        print(f"No se pudo generar '{input_file_for_knn}' porque '{base_train_merged_cleaned_file_name}' no está disponible.")
else:
    print(f"--- '{input_file_for_knn}' ya existe. ---")
    success_generating_input_knn_file = True
print("-" * 50)

# --- PASO 5b: Cargar datos para el modelo KNN ---
df_model_input = None
data_for_knn_analysis = pd.DataFrame()

if success_generating_input_knn_file and os.path.exists(input_file_for_knn):
    print(f"--- PASO 5b: Cargando datos para el modelo KNN desde '{input_file_for_knn}' ---")
    try:
        df_model_input = pd.read_csv(input_file_for_knn)
        print(f"Cargado '{input_file_for_knn}'. Forma: {df_model_input.shape}")
    except Exception as e:
        print(f"Error cargando '{input_file_for_knn}': {e}")
        df_model_input = None
else:
    print(f"No se puede proceder con el modelo KNN: '{input_file_for_knn}' no disponible.")
print("-" * 50)

# --- PASO 6: Preparación de datos para KNN ---
if df_model_input is not None and not df_model_input.empty:
    print(f"--- PASO 6: Preparando datos para KNN ---")

    target_col = 'price'

    # <<< INICIO MODIFICACIÓN: Crear 'property_age' y actualizar feature_cols >>>
    if 'yearbuilt' in df_model_input.columns:
        # Asegurarse de que 'yearbuilt' no tenga NaNs antes de la resta
        # El dropna en df_props debería haberlo manejado, pero una comprobación/fillna aquí podría ser más robusta
        # si la cadena de procesamiento anterior no lo garantizó completamente para df_model_input.
        # Por ahora, asumimos que 'yearbuilt' está limpio de NaNs en df_model_input debido al preprocesamiento.
        df_model_input['property_age'] = 2016 - df_model_input['yearbuilt']
        print("Característica 'property_age' creada y añadida a df_model_input.")
        # Definir feature_cols usando 'property_age'
        feature_cols = ['zip_price_rank', 'property_age', 'calculatedfinishedsquarefeet', 'bathroomcnt', 'bedroomcnt']
    else:
        print("Advertencia: La columna 'yearbuilt' no se encontró en df_model_input. 'property_age' no pudo ser creada.")
        print("Se utilizarán las features originales incluyendo 'yearbuilt' si está disponible.")
        # Fallback a las features originales si 'yearbuilt' no está para crear 'property_age'
        feature_cols = ['zip_price_rank', 'yearbuilt', 'calculatedfinishedsquarefeet']
    # <<< FIN MODIFICACIÓN >>>

    all_needed_cols = feature_cols + [target_col]
    missing_cols = [col for col in all_needed_cols if col not in df_model_input.columns]

    if missing_cols:
        print(f"Error: Faltan las siguientes columnas necesarias para el modelo KNN: {missing_cols}")
        df_model_input = None
    else:
        print(f"Features seleccionadas para el modelo: {feature_cols}")
        print(f"Target seleccionado: {target_col}")

        X = df_model_input[feature_cols]
        y = df_model_input[target_col]

        temp_df_for_nan_cleaning = X.copy()
        temp_df_for_nan_cleaning[target_col] = y

        rows_before_nan_drop = len(temp_df_for_nan_cleaning)
        # Asegurarse de que no haya NaNs en las columnas seleccionadas (incluyendo la nueva 'property_age')
        temp_df_for_nan_cleaning.dropna(subset=feature_cols + [target_col], inplace=True)
        rows_after_nan_drop = len(temp_df_for_nan_cleaning)
        print(f"Filas eliminadas por NaNs en features/target seleccionados: {rows_before_nan_drop - rows_after_nan_drop}")

        data_for_knn_analysis = temp_df_for_nan_cleaning

        if data_for_knn_analysis.empty:
            print("No quedan datos después de eliminar NaNs. No se puede continuar con el modelo KNN.")
            df_model_input = None
        else:
            X = data_for_knn_analysis[feature_cols]
            y = data_for_knn_analysis[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Datos divididos: {len(X_train)} entrenamiento ({len(X_train)/len(X)*100:.2f}%), {len(X_test)} prueba ({len(X_test)/len(X)*100:.2f}%).")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            print("Features escaladas.")

            # --- PASO 7: Análisis de `k` óptimo para KNN ---
            print("\n--- PASO 7: Analizando k óptimo para KNN Regressor (prediciendo 'price') ---")
            k_range = range(1, 31)
            mse_values = []

            print("Calculando MSE para diferentes valores de k (usando weights='distance')...")
            for k_val in k_range:
                # <<< MODIFICACIÓN: Añadir weights='distance' >>>
                knn_model = KNeighborsRegressor(n_neighbors=k_val, weights='distance')
                knn_model.fit(X_train_scaled, y_train)
                y_pred = knn_model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                mse_values.append(mse)
                # print(f"k = {k_val}, MSE = {mse:.2f}")

            plt.figure(figsize=(12, 6))
            plt.plot(k_range, mse_values, marker='o', linestyle='-')
            plt.title(f'MSE vs. Número de Vecinos (k) para KNN Regressor (Prediciendo {target_col}, weights=distance)')
            plt.xlabel('Número de Vecinos (k)')
            plt.ylabel('Mean Squared Error (MSE)')
            plt.xticks(list(k_range))
            plt.grid(True)
            plt.show()

            if mse_values:
                optimal_k_mse = min(mse_values)
                optimal_k_value = k_range[mse_values.index(optimal_k_mse)]
                print(f"\nEl MSE mínimo (con weights='distance') es {optimal_k_mse:.2f} para k = {optimal_k_value}.")
            else:
                print("\nNo se pudieron calcular los valores de MSE.")
            print("Busca en el gráfico un 'codo' o el punto donde el MSE deja de disminuir significativamente.")

            # --- PASO 8: Evaluación detallada con k elegido ---
            # El usuario mencionó k=22 como bueno en una configuración anterior.
            # Se evaluará con k=22 para esta nueva configuración.
            # El optimal_k_value de PASO 7 podría ser diferente ahora.
            print("\n--- PASO 8: Evaluación detallada con k elegido (weights='distance') ---")
            k_elegido = 22 # Manteniendo k=22 según la observación del usuario para comparar el efecto de los cambios

            print(f"Evaluando el modelo con k={k_elegido} y weights='distance'...")

            # 1. Entrenar el modelo KNN con k_elegido y weights='distance'
            # <<< MODIFICACIÓN: Añadir weights='distance' >>>
            knn_model_elegido = KNeighborsRegressor(n_neighbors=k_elegido, weights='distance')
            knn_model_elegido.fit(X_train_scaled, y_train)

            # 2. Realizar predicciones en el conjunto de prueba
            y_pred_elegido = knn_model_elegido.predict(X_test_scaled)

            # 3. Calcular MSE y RMSE para k_elegido
            mse_elegido = mean_squared_error(y_test, y_pred_elegido)
            rmse_elegido = np.sqrt(mse_elegido)
            print(f"  Mean Squared Error (MSE) para k={k_elegido}: {mse_elegido:.2f}")
            print(f"  Root Mean Squared Error (RMSE) para k={k_elegido}: {rmse_elegido:.2f} (error promedio en unidades de '{target_col}')")

            # 4. Calcular R-cuadrado (R²)
            r2_elegido = r2_score(y_test, y_pred_elegido)
            print(f"  Coeficiente de Determinación (R²) para k={k_elegido}: {r2_elegido:.4f}")
            print(f"  Esto significa que el modelo con k={k_elegido} explica aproximadamente el {r2_elegido*100:.2f}% de la variabilidad en '{target_col}'.")

            # 5. Calcular MAPE
            if np.all(y_test > 0):
                mape_elegido = mean_absolute_percentage_error(y_test, y_pred_elegido)
                print(f"  Error Porcentual Absoluto Medio (MAPE) para k={k_elegido}: {mape_elegido:.4f} ({mape_elegido*100:.2f}%)")
                print(f"  Esto significa que, en promedio, las predicciones del modelo con k={k_elegido} se desvían en un {mape_elegido*100:.2f}% del valor real de '{target_col}'.")
                if mape_elegido < 1:
                     print(f"  Podrías interpretar una 'efectividad porcentual' (basada en MAPE) como {(1-mape_elegido)*100:.2f}%.")
            else:
                print(f"  MAPE no se calculó o podría no ser fiable para k={k_elegido} porque '{target_col}' (y_test) contiene valores cero o negativos.")
                print(f"  Considera R² y RMSE como las principales métricas de rendimiento en este caso.")
            # --- FIN PASO 8 ---

else:
    if df_model_input is not None and df_model_input.empty :
         print("El DataFrame de entrada para el modelo KNN está vacío.")
    else: # df_model_input es None
         print("No se pudo cargar o generar el DataFrame de entrada para el modelo KNN.")

print("-" * 50)
# --- Resumen Final ---
if df_model_input is not None and not data_for_knn_analysis.empty:
    print("\n✅ Proceso de análisis de k para KNN completado.")
    if 'k_elegido' in locals() or 'optimal_k_value' in locals():
        # Determinar qué k se usó para la evaluación final en PASO 8
        k_final_eval_paso8 = k_elegido if 'k_elegido' in locals() and 'knn_model_elegido' in locals() else "no evaluado en Paso 8"
        print(f"Se evaluaron métricas adicionales (incluyendo R², weights='distance') para k={k_final_eval_paso8} en Paso 8.")
        if 'optimal_k_value' in locals():
             print(f"El análisis de k en Paso 7 (con weights='distance') sugirió k={optimal_k_value} como óptimo.")
    print(f"Se analizó k para predecir '{target_col}' usando las features: {feature_cols if 'feature_cols' in locals() else 'no definidas'}.") # Muestra las features usadas
else:
    print("\n❌ Proceso de análisis de k para KNN no pudo completarse debido a problemas con los datos.")

import pandas as pd
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Función auxiliar para ejecutar comandos de shell
def run_shell_command(command):
    print(f"Ejecutando: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout: print(result.stdout)
        if result.stderr: print(f"Shell Error (stderr): {result.stderr}") # Prefixed to distinguish
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar comando: {command}\nError: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        raise

# --- PASO 0: CONFIGURACIÓN DE KAGGLE ---
print("Configurando Kaggle CLI...")
if os.path.exists('/content/kaggle.json'):
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    run_shell_command(f"cp /content/kaggle.json {os.path.expanduser('~/.kaggle/')}")
    run_shell_command(f"chmod 600 {os.path.expanduser('~/.kaggle/kaggle.json')}")
    print("Archivo kaggle.json configurado.")
else:
    print("Advertencia: /content/kaggle.json no encontrado. Descarga podría fallar.")
print("-" * 50)

# --- PASO 1: Definiciones de archivos y parámetros ---
print("--- PASO 1: Definiciones ---")
original_properties_file_name = 'properties_2016.csv'
initial_cleaned_properties_file_name = 'properties_2016_initial_cleaned.csv'
original_train_file_name = 'train_2016_v2.csv'
base_train_merged_cleaned_file_name = 'train_final_cleaned_merged.csv'
# This is the file we need as input for the KNN model
input_file_for_knn = 'train_ranked_by_zip_price.csv'

kaggle_dataset_name = 'zillow-prize-1'

dtype_properties = {
    'parcelid': np.int32, 'bathroomcnt': np.float32, 'bedroomcnt': np.float32,
    'calculatedfinishedsquarefeet': np.float32, 'yearbuilt': np.float32,
    'taxvaluedollarcnt': np.float32, 'taxamount': np.float32,
    'latitude': np.float32, 'longitude': np.float32,
    'regionidzip': np.float32, 'regionidcounty': np.float32
}
selected_property_features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                              'yearbuilt', 'taxvaluedollarcnt', 'taxamount', 'latitude', 'longitude',
                              'regionidzip', 'regionidcounty']
dtype_train = {'parcelid': np.int32}

print(f"Archivo de entrada para KNN (debe existir o se generará): {input_file_for_knn}")
print("-" * 50)

# --- PASO 2, 3, 4, 5a: Generar/Cargar input_file_for_knn ---
success_generating_input_knn_file = False
if not os.path.exists(input_file_for_knn):
    print(f"--- '{input_file_for_knn}' no existe. Intentando generarlo (esto puede tardar)... ---")

    # Sub-Step A: Generate initial_cleaned_properties_file_name
    success_initial_cleaned_properties = False
    if not os.path.exists(initial_cleaned_properties_file_name):
        if not os.path.exists(original_properties_file_name):
            print(f"Descargando '{original_properties_file_name}'...")
            try:
                run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name}.zip --force")
                if os.path.exists(f'{original_properties_file_name}.zip'): run_shell_command(f"unzip -o {original_properties_file_name}.zip")
                if not os.path.exists(original_properties_file_name): run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name} --force")
                if not os.path.exists(original_properties_file_name): raise Exception(f"Fallo descarga {original_properties_file_name}")
            except Exception as e: print(f"Error descargando '{original_properties_file_name}': {e}")

        if os.path.exists(original_properties_file_name):
            try:
                print(f"Limpiando '{original_properties_file_name}'...")
                df_props = pd.read_csv(original_properties_file_name, usecols=selected_property_features, dtype=dtype_properties)
                rows_before = len(df_props)
                df_props.dropna(subset=['yearbuilt'], inplace=True) # Asegura que 'yearbuilt' no tenga NaNs antes de usarlo
                df_props = df_props[df_props['bathroomcnt'] != 0]
                df_props = df_props[df_props['bedroomcnt'] != 0]
                df_props.to_csv(initial_cleaned_properties_file_name, index=False)
                print(f"'{initial_cleaned_properties_file_name}' generado (Filas: {len(df_props)} de {rows_before}).")
                success_initial_cleaned_properties = True
            except Exception as e: print(f"Error generando '{initial_cleaned_properties_file_name}': {e}")
        else: print(f"'{original_properties_file_name}' no disponible para generar '{initial_cleaned_properties_file_name}'.")
    else:
        print(f"'{initial_cleaned_properties_file_name}' ya existe.")
        success_initial_cleaned_properties = True

    # Sub-Step B: Ensure original_train_file_name is available
    success_train_file_available = False
    if not os.path.exists(original_train_file_name):
        print(f"Descargando '{original_train_file_name}'...")
        try:
            run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_train_file_name} --force")
            if os.path.exists(original_train_file_name): success_train_file_available = True
            elif os.path.exists(f'{original_train_file_name}.zip'):
                run_shell_command(f"unzip -o {original_train_file_name}.zip")
                if os.path.exists(original_train_file_name): success_train_file_available = True
            if not success_train_file_available: raise Exception(f"Fallo descarga {original_train_file_name}")
        except Exception as e: print(f"Error descargando '{original_train_file_name}': {e}")
    else:
        print(f"'{original_train_file_name}' ya existe.")
        success_train_file_available = True

    # Sub-Step C: Generate base_train_merged_cleaned_file_name
    success_generating_base_merged_file = False
    if not os.path.exists(base_train_merged_cleaned_file_name):
        if success_initial_cleaned_properties and success_train_file_available:
            try:
                print(f"Cargando y uniendo '{initial_cleaned_properties_file_name}' y '{original_train_file_name}'...")
                df_props_cleaned = pd.read_csv(initial_cleaned_properties_file_name, dtype=dtype_properties)
                df_props_cleaned = df_props_cleaned[[col for col in selected_property_features if col in df_props_cleaned.columns]]

                df_train_orig = pd.read_csv(original_train_file_name, dtype=dtype_train, parse_dates=['transactiondate'])
                valid_parcelids = set(df_props_cleaned['parcelid'])
                df_train_ids_filtered = df_train_orig[df_train_orig['parcelid'].isin(valid_parcelids)]

                df_merged = pd.merge(df_train_ids_filtered, df_props_cleaned, on='parcelid', how='inner')

                if not df_merged.empty:
                    columns_to_clean_nans = ['regionidzip', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet'] # 'yearbuilt' ya no tiene NaNs
                    actual_columns_to_clean = [col for col in columns_to_clean_nans if col in df_merged.columns]
                    df_base_cleaned = df_merged.dropna(subset=actual_columns_to_clean) if actual_columns_to_clean else df_merged.copy()

                    if not df_base_cleaned.empty:
                        df_base_cleaned.to_csv(base_train_merged_cleaned_file_name, index=False)
                        print(f"'{base_train_merged_cleaned_file_name}' generado.")
                        success_generating_base_merged_file = True
                    else: print(f"'{base_train_merged_cleaned_file_name}' estaría vacío. No se guardó.")
                else: print("DataFrame unido vacío.")
            except Exception as e: print(f"Error generando '{base_train_merged_cleaned_file_name}': {e}")
        else: print(f"'{base_train_merged_cleaned_file_name}' no generado por falta de archivos base.")
    else:
        print(f"'{base_train_merged_cleaned_file_name}' ya existe.")
        success_generating_base_merged_file = True

    # Sub-Step D: Add 'price' and 'zip_price_rank' to create input_file_for_knn
    if success_generating_base_merged_file and os.path.exists(base_train_merged_cleaned_file_name):
        try:
            print(f"Creando '{input_file_for_knn}' a partir de '{base_train_merged_cleaned_file_name}'...")
            df_loaded_base = pd.read_csv(base_train_merged_cleaned_file_name)
            if not df_loaded_base.empty:
                df_to_rank = df_loaded_base.copy()
                if 'taxvaluedollarcnt' in df_to_rank.columns and 'taxamount' in df_to_rank.columns:
                    df_to_rank['price'] = df_to_rank['taxvaluedollarcnt'].fillna(0) - df_to_rank['taxamount'].fillna(0)
                else: raise ValueError("Columnas para 'price' no encontradas.")

                temp_df_for_median = df_to_rank.dropna(subset=['regionidzip', 'price'])
                if not temp_df_for_median.empty:
                    median_price_per_zip = temp_df_for_median.groupby('regionidzip')['price'].median().reset_index()
                    median_price_per_zip.rename(columns={'price': 'median_zip_price'}, inplace=True)
                    if not median_price_per_zip.empty:
                        median_price_per_zip['zip_price_rank'] = median_price_per_zip['median_zip_price'].rank(method='min', ascending=True).astype(int)
                        df_ranked = pd.merge(df_to_rank, median_price_per_zip[['regionidzip', 'zip_price_rank']], on='regionidzip', how='left')
                    else:
                        df_ranked = df_to_rank.copy()
                        df_ranked['zip_price_rank'] = np.nan
                else:
                    df_ranked = df_to_rank.copy()
                    df_ranked['zip_price_rank'] = np.nan

                df_ranked.to_csv(input_file_for_knn, index=False)
                print(f"'{input_file_for_knn}' generado y guardado.")
                success_generating_input_knn_file = True
            else: print(f"'{base_train_merged_cleaned_file_name}' está vacío.")
        except Exception as e: print(f"Error generando '{input_file_for_knn}': {e}")
    else:
        print(f"No se pudo generar '{input_file_for_knn}' porque '{base_train_merged_cleaned_file_name}' no está disponible.")
else:
    print(f"--- '{input_file_for_knn}' ya existe. ---")
    success_generating_input_knn_file = True
print("-" * 50)

# --- PASO 5b: Cargar datos para el modelo KNN ---
df_model_input = None
data_for_knn_analysis = pd.DataFrame()

if success_generating_input_knn_file and os.path.exists(input_file_for_knn):
    print(f"--- PASO 5b: Cargando datos para el modelo KNN desde '{input_file_for_knn}' ---")
    try:
        df_model_input = pd.read_csv(input_file_for_knn)
        print(f"Cargado '{input_file_for_knn}'. Forma: {df_model_input.shape}")
    except Exception as e:
        print(f"Error cargando '{input_file_for_knn}': {e}")
        df_model_input = None
else:
    print(f"No se puede proceder con el modelo KNN: '{input_file_for_knn}' no disponible.")
print("-" * 50)

# --- PASO 6: Preparación de datos para KNN ---
if df_model_input is not None and not df_model_input.empty:
    print(f"--- PASO 6: Preparando datos para KNN ---")

    target_col = 'taxvaluedollarcnt'

    # <<< INICIO MODIFICACIÓN: Crear 'property_age' y actualizar feature_cols >>>
    if 'yearbuilt' in df_model_input.columns:
        # Asegurarse de que 'yearbuilt' no tenga NaNs antes de la resta
        # El dropna en df_props debería haberlo manejado, pero una comprobación/fillna aquí podría ser más robusta
        # si la cadena de procesamiento anterior no lo garantizó completamente para df_model_input.
        # Por ahora, asumimos que 'yearbuilt' está limpio de NaNs en df_model_input debido al preprocesamiento.
        df_model_input['property_age'] = 2016 - df_model_input['yearbuilt']
        print("Característica 'property_age' creada y añadida a df_model_input.")
        # Definir feature_cols usando 'property_age'
        feature_cols = ['zip_price_rank', 'property_age', 'calculatedfinishedsquarefeet', 'bathroomcnt', 'bedroomcnt']
    else:
        print("Advertencia: La columna 'yearbuilt' no se encontró en df_model_input. 'property_age' no pudo ser creada.")
        print("Se utilizarán las features originales incluyendo 'yearbuilt' si está disponible.")
        # Fallback a las features originales si 'yearbuilt' no está para crear 'property_age'
        feature_cols = ['zip_price_rank', 'yearbuilt', 'calculatedfinishedsquarefeet']
    # <<< FIN MODIFICACIÓN >>>

    all_needed_cols = feature_cols + [target_col]
    missing_cols = [col for col in all_needed_cols if col not in df_model_input.columns]

    if missing_cols:
        print(f"Error: Faltan las siguientes columnas necesarias para el modelo KNN: {missing_cols}")
        df_model_input = None
    else:
        print(f"Features seleccionadas para el modelo: {feature_cols}")
        print(f"Target seleccionado: {target_col}")

        X = df_model_input[feature_cols]
        y = df_model_input[target_col]

        temp_df_for_nan_cleaning = X.copy()
        temp_df_for_nan_cleaning[target_col] = y

        rows_before_nan_drop = len(temp_df_for_nan_cleaning)
        # Asegurarse de que no haya NaNs en las columnas seleccionadas (incluyendo la nueva 'property_age')
        temp_df_for_nan_cleaning.dropna(subset=feature_cols + [target_col], inplace=True)
        rows_after_nan_drop = len(temp_df_for_nan_cleaning)
        print(f"Filas eliminadas por NaNs en features/target seleccionados: {rows_before_nan_drop - rows_after_nan_drop}")

        data_for_knn_analysis = temp_df_for_nan_cleaning

        if data_for_knn_analysis.empty:
            print("No quedan datos después de eliminar NaNs. No se puede continuar con el modelo KNN.")
            df_model_input = None
        else:
            X = data_for_knn_analysis[feature_cols]
            y = data_for_knn_analysis[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Datos divididos: {len(X_train)} entrenamiento ({len(X_train)/len(X)*100:.2f}%), {len(X_test)} prueba ({len(X_test)/len(X)*100:.2f}%).")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            print("Features escaladas.")

            # --- PASO 7: Análisis de `k` óptimo para KNN ---
            print("\n--- PASO 7: Analizando k óptimo para KNN Regressor (prediciendo 'price') ---")
            k_range = range(1, 31)
            mse_values = []

            print("Calculando MSE para diferentes valores de k (usando weights='distance')...")
            for k_val in k_range:
                # <<< MODIFICACIÓN: Añadir weights='distance' >>>
                knn_model = KNeighborsRegressor(n_neighbors=k_val)
                knn_model.fit(X_train_scaled, y_train)
                y_pred = knn_model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                mse_values.append(mse)
                # print(f"k = {k_val}, MSE = {mse:.2f}")

            plt.figure(figsize=(12, 6))
            plt.plot(k_range, mse_values, marker='o', linestyle='-')
            plt.title(f'MSE vs. Número de Vecinos (k) para KNN Regressor (Prediciendo {target_col}, weights=distance)')
            plt.xlabel('Número de Vecinos (k)')
            plt.ylabel('Mean Squared Error (MSE)')
            plt.xticks(list(k_range))
            plt.grid(True)
            plt.show()

            if mse_values:
                optimal_k_mse = min(mse_values)
                optimal_k_value = k_range[mse_values.index(optimal_k_mse)]
                print(f"\nEl MSE mínimo (con weights='distance') es {optimal_k_mse:.2f} para k = {optimal_k_value}.")
            else:
                print("\nNo se pudieron calcular los valores de MSE.")
            print("Busca en el gráfico un 'codo' o el punto donde el MSE deja de disminuir significativamente.")

            # --- PASO 8: Evaluación detallada con k elegido ---
            # El usuario mencionó k=22 como bueno en una configuración anterior.
            # Se evaluará con k=22 para esta nueva configuración.
            # El optimal_k_value de PASO 7 podría ser diferente ahora.
            print("\n--- PASO 8: Evaluación detallada con k elegido (weights='distance') ---")
            k_elegido = 22 # Manteniendo k=22 según la observación del usuario para comparar el efecto de los cambios

            print(f"Evaluando el modelo con k={k_elegido} y weights='distance'...")

            # 1. Entrenar el modelo KNN con k_elegido y weights='distance'
            # <<< MODIFICACIÓN: Añadir weights='distance' >>>
            knn_model_elegido = KNeighborsRegressor(n_neighbors=k_elegido)
            knn_model_elegido.fit(X_train_scaled, y_train)

            # 2. Realizar predicciones en el conjunto de prueba
            y_pred_elegido = knn_model_elegido.predict(X_test_scaled)

            # 3. Calcular MSE y RMSE para k_elegido
            mse_elegido = mean_squared_error(y_test, y_pred_elegido)
            rmse_elegido = np.sqrt(mse_elegido)
            print(f"  Mean Squared Error (MSE) para k={k_elegido}: {mse_elegido:.2f}")
            print(f"  Root Mean Squared Error (RMSE) para k={k_elegido}: {rmse_elegido:.2f} (error promedio en unidades de '{target_col}')")

            # 4. Calcular R-cuadrado (R²)
            r2_elegido = r2_score(y_test, y_pred_elegido)
            print(f"  Coeficiente de Determinación (R²) para k={k_elegido}: {r2_elegido:.4f}")
            print(f"  Esto significa que el modelo con k={k_elegido} explica aproximadamente el {r2_elegido*100:.2f}% de la variabilidad en '{target_col}'.")

            # 5. Calcular MAPE
            if np.all(y_test > 0):
                mape_elegido = mean_absolute_percentage_error(y_test, y_pred_elegido)
                print(f"  Error Porcentual Absoluto Medio (MAPE) para k={k_elegido}: {mape_elegido:.4f} ({mape_elegido*100:.2f}%)")
                print(f"  Esto significa que, en promedio, las predicciones del modelo con k={k_elegido} se desvían en un {mape_elegido*100:.2f}% del valor real de '{target_col}'.")
                if mape_elegido < 1:
                     print(f"  Podrías interpretar una 'efectividad porcentual' (basada en MAPE) como {(1-mape_elegido)*100:.2f}%.")
            else:
                print(f"  MAPE no se calculó o podría no ser fiable para k={k_elegido} porque '{target_col}' (y_test) contiene valores cero o negativos.")
                print(f"  Considera R² y RMSE como las principales métricas de rendimiento en este caso.")

            # 6. Graficar Valores Reales vs. Predichos
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred_elegido, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2) # Línea de referencia y=x
            plt.xlabel("Valores Reales (price)")
            plt.ylabel("Valores Predichos (price)")
            plt.title(f"Valores Reales vs. Predichos para k={k_elegido} (weights='distance')")
            plt.grid(True)
            plt.show()
            # --- FIN PASO 8 ---

else:
    if df_model_input is not None and df_model_input.empty :
         print("El DataFrame de entrada para el modelo KNN está vacío.")
    else: # df_model_input es None
         print("No se pudo cargar o generar el DataFrame de entrada para el modelo KNN.")

print("-" * 50)
# --- Resumen Final ---
if df_model_input is not None and not data_for_knn_analysis.empty:
    print("\n✅ Proceso de análisis de k para KNN completado.")
    if 'k_elegido' in locals() or 'optimal_k_value' in locals():
        # Determinar qué k se usó para la evaluación final en PASO 8
        k_final_eval_paso8 = k_elegido if 'k_elegido' in locals() and 'knn_model_elegido' in locals() else "no evaluado en Paso 8"
        print(f"Se evaluaron métricas adicionales (incluyendo R², weights='distance') para k={k_final_eval_paso8} en Paso 8.")
        if 'optimal_k_value' in locals():
             print(f"El análisis de k en Paso 7 (con weights='distance') sugirió k={optimal_k_value} como óptimo.")
    print(f"Se analizó k para predecir '{target_col}' usando las features: {feature_cols if 'feature_cols' in locals() else 'no definidas'}.") # Muestra las features usadas
else:
    print("\n❌ Proceso de análisis de k para KNN no pudo completarse debido a problemas con los datos.")
import pandas as pd
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Función auxiliar para ejecutar comandos de shell
def run_shell_command(command):
    print(f"Ejecutando: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout: print(result.stdout)
        if result.stderr: print(f"Shell Error (stderr): {result.stderr}") # Prefixed to distinguish
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar comando: {command}\nError: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        raise

# --- PASO 0: CONFIGURACIÓN DE KAGGLE ---
print("Configurando Kaggle CLI...")
if os.path.exists('/content/kaggle.json'):
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    run_shell_command(f"cp /content/kaggle.json {os.path.expanduser('~/.kaggle/')}")
    run_shell_command(f"chmod 600 {os.path.expanduser('~/.kaggle/kaggle.json')}")
    print("Archivo kaggle.json configurado.")
else:
    print("Advertencia: /content/kaggle.json no encontrado. Descarga podría fallar.")
print("-" * 50)

# --- PASO 1: Definiciones de archivos y parámetros ---
print("--- PASO 1: Definiciones ---")
original_properties_file_name = 'properties_2016.csv'
initial_cleaned_properties_file_name = 'properties_2016_initial_cleaned.csv'
original_train_file_name = 'train_2016_v2.csv'
base_train_merged_cleaned_file_name = 'train_final_cleaned_merged.csv'
input_file_for_knn = 'train_ranked_by_zip_price.csv' # Se regenerará si no existe

kaggle_dataset_name = 'zillow-prize-1'

dtype_properties = {
    'parcelid': np.int32, 'bathroomcnt': np.float32, 'bedroomcnt': np.float32,
    'calculatedfinishedsquarefeet': np.float32, 'yearbuilt': np.float32,
    'taxvaluedollarcnt': np.float32, 'taxamount': np.float32,
    'latitude': np.float32, 'longitude': np.float32,
    'regionidzip': np.float32, 'regionidcounty': np.float32
}
selected_property_features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                              'yearbuilt', 'taxvaluedollarcnt', 'taxamount', 'latitude', 'longitude',
                              'regionidzip', 'regionidcounty']
dtype_train = {'parcelid': np.int32}

print(f"Archivo de entrada para KNN (debe existir o se generará): {input_file_for_knn}")
print("NOTA: Si deseas regenerar este archivo con las nuevas columnas de ranking, elimínalo antes de ejecutar.")
print("-" * 50)

# --- PASO 2, 3, 4, 5a: Generar/Cargar input_file_for_knn ---
success_generating_input_knn_file = False
if not os.path.exists(input_file_for_knn):
    print(f"--- '{input_file_for_knn}' no existe. Intentando generarlo (esto puede tardar)... ---")

    # Sub-Step A (igual que antes)
    success_initial_cleaned_properties = False
    if not os.path.exists(initial_cleaned_properties_file_name):
        if not os.path.exists(original_properties_file_name):
            print(f"Descargando '{original_properties_file_name}'...")
            try:
                run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name}.zip --force")
                if os.path.exists(f'{original_properties_file_name}.zip'): run_shell_command(f"unzip -o {original_properties_file_name}.zip")
                if not os.path.exists(original_properties_file_name): run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name} --force")
                if not os.path.exists(original_properties_file_name): raise Exception(f"Fallo descarga {original_properties_file_name}")
            except Exception as e: print(f"Error descargando '{original_properties_file_name}': {e}")

        if os.path.exists(original_properties_file_name):
            try:
                print(f"Limpiando '{original_properties_file_name}'...")
                df_props = pd.read_csv(original_properties_file_name, usecols=selected_property_features, dtype=dtype_properties)
                rows_before = len(df_props)
                df_props.dropna(subset=['yearbuilt'], inplace=True)
                df_props = df_props[df_props['bathroomcnt'] != 0]
                df_props = df_props[df_props['bedroomcnt'] != 0]
                df_props.to_csv(initial_cleaned_properties_file_name, index=False)
                print(f"'{initial_cleaned_properties_file_name}' generado (Filas: {len(df_props)} de {rows_before}).")
                success_initial_cleaned_properties = True
            except Exception as e: print(f"Error generando '{initial_cleaned_properties_file_name}': {e}")
        else: print(f"'{original_properties_file_name}' no disponible para generar '{initial_cleaned_properties_file_name}'.")
    else:
        print(f"'{initial_cleaned_properties_file_name}' ya existe.")
        success_initial_cleaned_properties = True

    # Sub-Step B (igual que antes)
    success_train_file_available = False
    if not os.path.exists(original_train_file_name):
        print(f"Descargando '{original_train_file_name}'...")
        try:
            run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_train_file_name} --force")
            if os.path.exists(original_train_file_name): success_train_file_available = True
            elif os.path.exists(f'{original_train_file_name}.zip'):
                run_shell_command(f"unzip -o {original_train_file_name}.zip")
                if os.path.exists(original_train_file_name): success_train_file_available = True
            if not success_train_file_available: raise Exception(f"Fallo descarga {original_train_file_name}")
        except Exception as e: print(f"Error descargando '{original_train_file_name}': {e}")
    else:
        print(f"'{original_train_file_name}' ya existe.")
        success_train_file_available = True

    # Sub-Step C (igual que antes)
    success_generating_base_merged_file = False
    if not os.path.exists(base_train_merged_cleaned_file_name):
        if success_initial_cleaned_properties and success_train_file_available:
            try:
                print(f"Cargando y uniendo '{initial_cleaned_properties_file_name}' y '{original_train_file_name}'...")
                df_props_cleaned = pd.read_csv(initial_cleaned_properties_file_name, dtype=dtype_properties)
                df_props_cleaned = df_props_cleaned[[col for col in selected_property_features if col in df_props_cleaned.columns]]

                df_train_orig = pd.read_csv(original_train_file_name, dtype=dtype_train, parse_dates=['transactiondate'])
                valid_parcelids = set(df_props_cleaned['parcelid'])
                df_train_ids_filtered = df_train_orig[df_train_orig['parcelid'].isin(valid_parcelids)]

                df_merged = pd.merge(df_train_ids_filtered, df_props_cleaned, on='parcelid', how='inner')

                if not df_merged.empty:
                    columns_to_clean_nans = ['regionidzip', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet']
                    actual_columns_to_clean = [col for col in columns_to_clean_nans if col in df_merged.columns]
                    df_base_cleaned = df_merged.dropna(subset=actual_columns_to_clean) if actual_columns_to_clean else df_merged.copy()

                    if not df_base_cleaned.empty:
                        df_base_cleaned.to_csv(base_train_merged_cleaned_file_name, index=False)
                        print(f"'{base_train_merged_cleaned_file_name}' generado.")
                        success_generating_base_merged_file = True
                    else: print(f"'{base_train_merged_cleaned_file_name}' estaría vacío. No se guardó.")
                else: print("DataFrame unido vacío.")
            except Exception as e: print(f"Error generando '{base_train_merged_cleaned_file_name}': {e}")
        else: print(f"'{base_train_merged_cleaned_file_name}' no generado por falta de archivos base.")
    else:
        print(f"'{base_train_merged_cleaned_file_name}' ya existe.")
        success_generating_base_merged_file = True

    # Sub-Step D: Add 'price' and RANKING COLUMNS to create input_file_for_knn
    if success_generating_base_merged_file and os.path.exists(base_train_merged_cleaned_file_name):
        try:
            print(f"Creando '{input_file_for_knn}' a partir de '{base_train_merged_cleaned_file_name}'...")
            df_loaded_base = pd.read_csv(base_train_merged_cleaned_file_name)
            if not df_loaded_base.empty:
                df_ranked = df_loaded_base.copy() # Empezar con una copia
                if 'taxvaluedollarcnt' in df_ranked.columns and 'taxamount' in df_ranked.columns:
                    df_ranked['price'] = df_ranked['taxvaluedollarcnt'].fillna(0) - df_ranked['taxamount'].fillna(0)
                else: raise ValueError("Columnas para 'price' no encontradas.")

                # Inicializar columnas de ranking
                df_ranked['zip_price_rank_mean'] = np.nan
                df_ranked['zip_price_rank_median'] = np.nan

                temp_df_for_aggregation = df_ranked.dropna(subset=['regionidzip', 'price'])

                if not temp_df_for_aggregation.empty:
                    # --- Calcular RANKING BASADO EN LA MEDIA ---
                    print("Calculando 'zip_price_rank_mean' (basado en precio PROMEDIO por regionidzip)...")
                    mean_price_per_zip = temp_df_for_aggregation.groupby('regionidzip')['price'].mean().reset_index()
                    mean_price_per_zip.rename(columns={'price': 'mean_zip_price_val'}, inplace=True)
                    if not mean_price_per_zip.empty:
                        mean_price_per_zip['zip_price_rank_mean'] = mean_price_per_zip['mean_zip_price_val'].rank(method='min', ascending=True).astype(int)
                        # Eliminar la columna de inicialización antes de unir para evitar duplicados de merge
                        df_ranked = df_ranked.drop(columns=['zip_price_rank_mean'], errors='ignore')
                        df_ranked = pd.merge(df_ranked, mean_price_per_zip[['regionidzip', 'zip_price_rank_mean']], on='regionidzip', how='left')
                        print("Columna 'zip_price_rank_mean' añadida.")

                    # --- Calcular RANKING BASADO EN LA MEDIANA ---
                    print("Calculando 'zip_price_rank_median' (basado en precio MEDIANO por regionidzip)...")
                    median_price_per_zip = temp_df_for_aggregation.groupby('regionidzip')['price'].median().reset_index()
                    median_price_per_zip.rename(columns={'price': 'median_zip_price_val'}, inplace=True)
                    if not median_price_per_zip.empty:
                        median_price_per_zip['zip_price_rank_median'] = median_price_per_zip['median_zip_price_val'].rank(method='min', ascending=True).astype(int)
                        # Eliminar la columna de inicialización antes de unir
                        df_ranked = df_ranked.drop(columns=['zip_price_rank_median'], errors='ignore')
                        df_ranked = pd.merge(df_ranked, median_price_per_zip[['regionidzip', 'zip_price_rank_median']], on='regionidzip', how='left')
                        print("Columna 'zip_price_rank_median' añadida.")
                else:
                    print("DataFrame para agregación vacío después de dropna. Columnas de ranking serán NaN.")

                df_ranked.to_csv(input_file_for_knn, index=False)
                print(f"'{input_file_for_knn}' generado y guardado con ambas columnas de ranking.")
                success_generating_input_knn_file = True
            else: print(f"'{base_train_merged_cleaned_file_name}' está vacío.")
        except Exception as e: print(f"Error generando '{input_file_for_knn}': {e}")
    else:
        print(f"No se pudo generar '{input_file_for_knn}' porque '{base_train_merged_cleaned_file_name}' no está disponible.")
else:
    print(f"--- '{input_file_for_knn}' ya existe. Se usará este archivo. ---")
    print(f"--- Asegúrate de que contenga 'zip_price_rank_mean' y 'zip_price_rank_median' o elimínalo para regenerarlo. ---")
    success_generating_input_knn_file = True
print("-" * 50)

# --- PASO 5b: Cargar datos para el modelo KNN ---
df_model_input = None
data_for_knn_analysis = pd.DataFrame()

if success_generating_input_knn_file and os.path.exists(input_file_for_knn):
    print(f"--- PASO 5b: Cargando datos para el modelo KNN desde '{input_file_for_knn}' ---")
    try:
        df_model_input = pd.read_csv(input_file_for_knn)
        print(f"Cargado '{input_file_for_knn}'. Forma: {df_model_input.shape}")
        # print("Columnas disponibles en df_model_input:", df_model_input.columns.tolist()) # Descomentar para depurar
    except Exception as e:
        print(f"Error cargando '{input_file_for_knn}': {e}")
        df_model_input = None
else:
    print(f"No se puede proceder con el modelo KNN: '{input_file_for_knn}' no disponible.")
print("-" * 50)

# --- PASO 6: Preparación de datos para KNN ---
if df_model_input is not None and not df_model_input.empty:
    print(f"--- PASO 6: Preparando datos para KNN ---")

    target_col = 'price'

    if 'yearbuilt' in df_model_input.columns:
        df_model_input['property_age'] = 2016 - df_model_input['yearbuilt']
        print("Característica 'property_age' creada y añadida a df_model_input.")
        # <<< MODIFICACIÓN: Usar 'zip_price_rank_median' y asegurar otras features correctas >>>
        feature_cols = ['zip_price_rank_mean', 'property_age', 'calculatedfinishedsquarefeet', 'bathroomcnt', 'bedroomcnt']
    else:
        print("Advertencia: La columna 'yearbuilt' no se encontró en df_model_input. 'property_age' no pudo ser creada.")
        print("Se utilizarán las features originales incluyendo 'yearbuilt' si está disponible.")
        # <<< MODIFICACIÓN: Usar 'zip_price_rank_median' y asegurar otras features correctas >>>
        feature_cols = ['zip_price_rank_mean', 'yearbuilt', 'calculatedfinishedsquarefeet', 'bathroomcnt', 'bedroomcnt']

    all_needed_cols = feature_cols + [target_col]
    missing_cols = [col for col in all_needed_cols if col not in df_model_input.columns]

    if missing_cols:
        print(f"Error: Faltan las siguientes columnas necesarias para el modelo KNN: {missing_cols}")
        print(f"Columnas disponibles en df_model_input: {df_model_input.columns.tolist()}")
        df_model_input = None
    else:
        print(f"Features seleccionadas para el modelo: {feature_cols}")
        print(f"Target seleccionado: {target_col}")

        X = df_model_input[feature_cols]
        y = df_model_input[target_col]

        temp_df_for_nan_cleaning = X.copy()
        temp_df_for_nan_cleaning[target_col] = y

        rows_before_nan_drop = len(temp_df_for_nan_cleaning)
        temp_df_for_nan_cleaning.dropna(subset=feature_cols + [target_col], inplace=True)
        rows_after_nan_drop = len(temp_df_for_nan_cleaning)
        print(f"Filas eliminadas por NaNs en features/target seleccionados: {rows_before_nan_drop - rows_after_nan_drop}")

        data_for_knn_analysis = temp_df_for_nan_cleaning

        if data_for_knn_analysis.empty:
            print("No quedan datos después de eliminar NaNs. No se puede continuar con el modelo KNN.")
            df_model_input = None
        else:
            X = data_for_knn_analysis[feature_cols]
            y = data_for_knn_analysis[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Datos divididos: {len(X_train)} entrenamiento ({len(X_train)/len(X)*100:.2f}%), {len(X_test)} prueba ({len(X_test)/len(X)*100:.2f}%).")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            print("Features escaladas.")

            # --- PASO 7: Análisis de `k` óptimo para KNN ---
            print(f"\n--- PASO 7: Analizando k óptimo para KNN Regressor (prediciendo '{target_col}') ---")
            k_range = range(1, 31)
            mse_values = []

            # <<< MODIFICACIÓN: Volver a weights='uniform' (comportamiento por defecto) >>>
            print("Calculando MSE para diferentes valores de k (usando weights='uniform')...")
            for k_val in k_range:
                knn_model = KNeighborsRegressor(n_neighbors=k_val) # weights='uniform' por defecto
                knn_model.fit(X_train_scaled, y_train)
                y_pred = knn_model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                mse_values.append(mse)

            plt.figure(figsize=(12, 6))
            plt.plot(k_range, mse_values, marker='o', linestyle='-')
            # <<< MODIFICACIÓN: Actualizar título del gráfico >>>
            plt.title(f'MSE vs. Número de Vecinos (k) para KNN Regressor (Prediciendo {target_col}, weights=uniform)')
            plt.xlabel('Número de Vecinos (k)')
            plt.ylabel('Mean Squared Error (MSE)')
            plt.xticks(list(k_range))
            plt.grid(True)
            plt.show()

            if mse_values:
                optimal_k_mse = min(mse_values)
                optimal_k_value = k_range[mse_values.index(optimal_k_mse)]
                # <<< MODIFICACIÓN: Actualizar print statement >>>
                print(f"\nEl MSE mínimo (con weights='uniform') es {optimal_k_mse:.2f} para k = {optimal_k_value}.")
            else:
                print("\nNo se pudieron calcular los valores de MSE.")
            print("Busca en el gráfico un 'codo' o el punto donde el MSE deja de disminuir significativamente.")

            # --- PASO 8: Evaluación detallada con k elegido ---
            print(f"\n--- PASO 8: Evaluación detallada con k elegido (weights='uniform', prediciendo '{target_col}') ---")
            k_elegido = 22

            # <<< MODIFICACIÓN: Volver a weights='uniform' (comportamiento por defecto) >>>
            print(f"Evaluando el modelo con k={k_elegido} y weights='uniform'...")
            knn_model_elegido = KNeighborsRegressor(n_neighbors=k_elegido) # weights='uniform' por defecto
            knn_model_elegido.fit(X_train_scaled, y_train)

            y_pred_elegido = knn_model_elegido.predict(X_test_scaled)

            mse_elegido = mean_squared_error(y_test, y_pred_elegido)
            rmse_elegido = np.sqrt(mse_elegido)
            print(f"  Mean Squared Error (MSE) para k={k_elegido}: {mse_elegido:.2f}")
            print(f"  Root Mean Squared Error (RMSE) para k={k_elegido}: {rmse_elegido:.2f} (error promedio en unidades de '{target_col}')")

            r2_elegido = r2_score(y_test, y_pred_elegido)
            print(f"  Coeficiente de Determinación (R²) para k={k_elegido}: {r2_elegido:.4f}")
            print(f"  Esto significa que el modelo con k={k_elegido} explica aproximadamente el {r2_elegido*100:.2f}% de la variabilidad en '{target_col}'.")

            if np.all(y_test > 0):
                mape_elegido = mean_absolute_percentage_error(y_test, y_pred_elegido)
                print(f"  Error Porcentual Absoluto Medio (MAPE) para k={k_elegido}: {mape_elegido:.4f} ({mape_elegido*100:.2f}%)")
                print(f"  Esto significa que, en promedio, las predicciones del modelo con k={k_elegido} se desvían en un {mape_elegido*100:.2f}% del valor real de '{target_col}'.")
                if mape_elegido < 1:
                     print(f"  Podrías interpretar una 'efectividad porcentual' (basada en MAPE) como {(1-mape_elegido)*100:.2f}%.")
            else:
                print(f"  MAPE no se calculó o podría no ser fiable para k={k_elegido} porque '{target_col}' (y_test) contiene valores cero o negativos.")
                print(f"  Considera R² y RMSE como las principales métricas de rendimiento en este caso.")

            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred_elegido, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
            plt.xlabel(f"Valores Reales ({target_col})")
            plt.ylabel(f"Valores Predichos ({target_col})")
            # <<< MODIFICACIÓN: Actualizar título del gráfico >>>
            plt.title(f"Valores Reales vs. Predichos para k={k_elegido} (weights='uniform')")
            plt.grid(True)
            plt.show()

else:
    if df_model_input is not None and df_model_input.empty :
         print("El DataFrame de entrada para el modelo KNN está vacío.")
    else: # df_model_input es None
         print("No se pudo cargar o generar el DataFrame de entrada para el modelo KNN.")

print("-" * 50)
# --- Resumen Final ---
if df_model_input is not None and not data_for_knn_analysis.empty:
    print("\n✅ Proceso de análisis de k para KNN completado.")
    if 'k_elegido' in locals() or 'optimal_k_value' in locals():
        k_final_eval_paso8 = k_elegido if 'k_elegido' in locals() and 'knn_model_elegido' in locals() else "no evaluado en Paso 8"
        # <<< MODIFICACIÓN: Actualizar print statement >>>
        print(f"Se evaluaron métricas adicionales (incluyendo R², weights='uniform') para k={k_final_eval_paso8} en Paso 8.")
        if 'optimal_k_value' in locals():
             print(f"El análisis de k en Paso 7 (con weights='uniform') sugirió k={optimal_k_value} como óptimo.")
    print(f"Se analizó k para predecir '{target_col}' usando las features: {feature_cols if 'feature_cols' in locals() else 'no definidas'}.")
else:
    print("\n❌ Proceso de análisis de k para KNN no pudo completarse debido a problemas con los datos.")
import pandas as pd
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Función auxiliar para ejecutar comandos de shell
def run_shell_command(command):
    print(f"Ejecutando: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout: print(result.stdout)
        if result.stderr: print(f"Shell Error (stderr): {result.stderr}") # Prefixed to distinguish
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar comando: {command}\nError: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        raise

# --- PASO 0: CONFIGURACIÓN DE KAGGLE ---
print("Configurando Kaggle CLI...")
if os.path.exists('/content/kaggle.json'):
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    run_shell_command(f"cp /content/kaggle.json {os.path.expanduser('~/.kaggle/')}")
    run_shell_command(f"chmod 600 {os.path.expanduser('~/.kaggle/kaggle.json')}")
    print("Archivo kaggle.json configurado.")
else:
    print("Advertencia: /content/kaggle.json no encontrado. Descarga podría fallar.")
print("-" * 50)

# --- PASO 1: Definiciones de archivos y parámetros ---
print("--- PASO 1: Definiciones ---")
original_properties_file_name = 'properties_2016.csv'
initial_cleaned_properties_file_name = 'properties_2016_initial_cleaned.csv'
original_train_file_name = 'train_2016_v2.csv'
base_train_merged_cleaned_file_name = 'train_final_cleaned_merged.csv'
input_file_for_knn = 'train_ranked_by_zip_price.csv'

kaggle_dataset_name = 'zillow-prize-1'

dtype_properties = {
    'parcelid': np.int32, 'bathroomcnt': np.float32, 'bedroomcnt': np.float32,
    'calculatedfinishedsquarefeet': np.float32, 'yearbuilt': np.float32,
    'taxvaluedollarcnt': np.float32, 'taxamount': np.float32,
    'latitude': np.float32, 'longitude': np.float32,
    'regionidzip': np.float32, 'regionidcounty': np.float32
}
selected_property_features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                              'yearbuilt', 'taxvaluedollarcnt', 'taxamount', 'latitude', 'longitude',
                              'regionidzip', 'regionidcounty']
dtype_train = {'parcelid': np.int32}

print(f"Archivo de entrada para KNN (debe existir o se generará): {input_file_for_knn}")
print("NOTA: Si deseas regenerar este archivo con las nuevas columnas de ranking, elimínalo antes de ejecutar.")
print("-" * 50)

# --- PASO 2, 3, 4, 5a: Generar/Cargar input_file_for_knn ---
success_generating_input_knn_file = False
if not os.path.exists(input_file_for_knn):
    print(f"--- '{input_file_for_knn}' no existe. Intentando generarlo (esto puede tardar)... ---")

    success_initial_cleaned_properties = False
    if not os.path.exists(initial_cleaned_properties_file_name):
        if not os.path.exists(original_properties_file_name):
            print(f"Descargando '{original_properties_file_name}'...")
            try:
                run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name}.zip --force")
                if os.path.exists(f'{original_properties_file_name}.zip'): run_shell_command(f"unzip -o {original_properties_file_name}.zip")
                if not os.path.exists(original_properties_file_name): run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_properties_file_name} --force")
                if not os.path.exists(original_properties_file_name): raise Exception(f"Fallo descarga {original_properties_file_name}")
            except Exception as e: print(f"Error descargando '{original_properties_file_name}': {e}")

        if os.path.exists(original_properties_file_name):
            try:
                print(f"Limpiando '{original_properties_file_name}'...")
                df_props = pd.read_csv(original_properties_file_name, usecols=selected_property_features, dtype=dtype_properties)
                rows_before = len(df_props)
                df_props.dropna(subset=['yearbuilt'], inplace=True)
                df_props = df_props[df_props['bathroomcnt'] != 0]
                df_props = df_props[df_props['bedroomcnt'] != 0]
                df_props.to_csv(initial_cleaned_properties_file_name, index=False)
                print(f"'{initial_cleaned_properties_file_name}' generado (Filas: {len(df_props)} de {rows_before}).")
                success_initial_cleaned_properties = True
            except Exception as e: print(f"Error generando '{initial_cleaned_properties_file_name}': {e}")
        else: print(f"'{original_properties_file_name}' no disponible para generar '{initial_cleaned_properties_file_name}'.")
    else:
        print(f"'{initial_cleaned_properties_file_name}' ya existe.")
        success_initial_cleaned_properties = True

    success_train_file_available = False
    if not os.path.exists(original_train_file_name):
        print(f"Descargando '{original_train_file_name}'...")
        try:
            run_shell_command(f"kaggle competitions download -c {kaggle_dataset_name} -f {original_train_file_name} --force")
            if os.path.exists(original_train_file_name): success_train_file_available = True
            elif os.path.exists(f'{original_train_file_name}.zip'):
                run_shell_command(f"unzip -o {original_train_file_name}.zip")
                if os.path.exists(original_train_file_name): success_train_file_available = True
            if not success_train_file_available: raise Exception(f"Fallo descarga {original_train_file_name}")
        except Exception as e: print(f"Error descargando '{original_train_file_name}': {e}")
    else:
        print(f"'{original_train_file_name}' ya existe.")
        success_train_file_available = True

    success_generating_base_merged_file = False
    if not os.path.exists(base_train_merged_cleaned_file_name):
        if success_initial_cleaned_properties and success_train_file_available:
            try:
                print(f"Cargando y uniendo '{initial_cleaned_properties_file_name}' y '{original_train_file_name}'...")
                df_props_cleaned = pd.read_csv(initial_cleaned_properties_file_name, dtype=dtype_properties)
                df_props_cleaned = df_props_cleaned[[col for col in selected_property_features if col in df_props_cleaned.columns]]

                df_train_orig = pd.read_csv(original_train_file_name, dtype=dtype_train, parse_dates=['transactiondate'])
                valid_parcelids = set(df_props_cleaned['parcelid'])
                df_train_ids_filtered = df_train_orig[df_train_orig['parcelid'].isin(valid_parcelids)]

                df_merged = pd.merge(df_train_ids_filtered, df_props_cleaned, on='parcelid', how='inner')

                if not df_merged.empty:
                    columns_to_clean_nans = ['regionidzip', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet']
                    actual_columns_to_clean = [col for col in columns_to_clean_nans if col in df_merged.columns]
                    df_base_cleaned = df_merged.dropna(subset=actual_columns_to_clean) if actual_columns_to_clean else df_merged.copy()

                    if not df_base_cleaned.empty:
                        df_base_cleaned.to_csv(base_train_merged_cleaned_file_name, index=False)
                        print(f"'{base_train_merged_cleaned_file_name}' generado.")
                        success_generating_base_merged_file = True
                    else: print(f"'{base_train_merged_cleaned_file_name}' estaría vacío. No se guardó.")
                else: print("DataFrame unido vacío.")
            except Exception as e: print(f"Error generando '{base_train_merged_cleaned_file_name}': {e}")
        else: print(f"'{base_train_merged_cleaned_file_name}' no generado por falta de archivos base.")
    else:
        print(f"'{base_train_merged_cleaned_file_name}' ya existe.")
        success_generating_base_merged_file = True

    if success_generating_base_merged_file and os.path.exists(base_train_merged_cleaned_file_name):
        try:
            print(f"Creando '{input_file_for_knn}' a partir de '{base_train_merged_cleaned_file_name}'...")
            df_loaded_base = pd.read_csv(base_train_merged_cleaned_file_name)
            if not df_loaded_base.empty:
                df_ranked = df_loaded_base.copy()
                if 'taxvaluedollarcnt' in df_ranked.columns and 'taxamount' in df_ranked.columns:
                    df_ranked['price'] = df_ranked['taxvaluedollarcnt'].fillna(0) - df_ranked['taxamount'].fillna(0)
                else: raise ValueError("Columnas para 'price' no encontradas.")

                df_ranked['zip_price_rank_mean'] = np.nan
                df_ranked['zip_price_rank_median'] = np.nan
                temp_df_for_aggregation = df_ranked.dropna(subset=['regionidzip', 'price'])

                if not temp_df_for_aggregation.empty:
                    print("Calculando 'zip_price_rank_mean' (basado en precio PROMEDIO por regionidzip)...")
                    mean_price_per_zip = temp_df_for_aggregation.groupby('regionidzip')['price'].mean().reset_index()
                    mean_price_per_zip.rename(columns={'price': 'mean_zip_price_val'}, inplace=True)
                    if not mean_price_per_zip.empty:
                        mean_price_per_zip['zip_price_rank_mean'] = mean_price_per_zip['mean_zip_price_val'].rank(method='min', ascending=True).astype(int)
                        df_ranked = df_ranked.drop(columns=['zip_price_rank_mean'], errors='ignore')
                        df_ranked = pd.merge(df_ranked, mean_price_per_zip[['regionidzip', 'zip_price_rank_mean']], on='regionidzip', how='left')
                        print("Columna 'zip_price_rank_mean' añadida.")

                    print("Calculando 'zip_price_rank_median' (basado en precio MEDIANO por regionidzip)...")
                    median_price_per_zip = temp_df_for_aggregation.groupby('regionidzip')['price'].median().reset_index()
                    median_price_per_zip.rename(columns={'price': 'median_zip_price_val'}, inplace=True)
                    if not median_price_per_zip.empty:
                        median_price_per_zip['zip_price_rank_median'] = median_price_per_zip['median_zip_price_val'].rank(method='min', ascending=True).astype(int)
                        df_ranked = df_ranked.drop(columns=['zip_price_rank_median'], errors='ignore')
                        df_ranked = pd.merge(df_ranked, median_price_per_zip[['regionidzip', 'zip_price_rank_median']], on='regionidzip', how='left')
                        print("Columna 'zip_price_rank_median' añadida.")
                else:
                    print("DataFrame para agregación vacío después de dropna. Columnas de ranking serán NaN.")

                df_ranked.to_csv(input_file_for_knn, index=False)
                print(f"'{input_file_for_knn}' generado y guardado con ambas columnas de ranking.")
                success_generating_input_knn_file = True
            else: print(f"'{base_train_merged_cleaned_file_name}' está vacío.")
        except Exception as e: print(f"Error generando '{input_file_for_knn}': {e}")
    else:
        print(f"No se pudo generar '{input_file_for_knn}' porque '{base_train_merged_cleaned_file_name}' no está disponible.")
else:
    print(f"--- '{input_file_for_knn}' ya existe. Se usará este archivo. ---")
    print(f"--- Asegúrate de que contenga 'zip_price_rank_mean' y 'zip_price_rank_median' o elimínalo para regenerarlo. ---")
    success_generating_input_knn_file = True
print("-" * 50)

# --- PASO 5b: Cargar datos para el modelo KNN ---
df_model_input = None
data_for_knn_analysis = pd.DataFrame()

if success_generating_input_knn_file and os.path.exists(input_file_for_knn):
    print(f"--- PASO 5b: Cargando datos para el modelo KNN desde '{input_file_for_knn}' ---")
    try:
        df_model_input = pd.read_csv(input_file_for_knn)
        print(f"Cargado '{input_file_for_knn}'. Forma: {df_model_input.shape}")
    except Exception as e:
        print(f"Error cargando '{input_file_for_knn}': {e}")
        df_model_input = None
else:
    print(f"No se puede proceder con el modelo KNN: '{input_file_for_knn}' no disponible.")
print("-" * 50)

# --- PASO 6: Preparación de datos para KNN ---
if df_model_input is not None and not df_model_input.empty:
    print(f"--- PASO 6: Preparando datos para KNN ---")
    target_col = 'price'

    if 'yearbuilt' in df_model_input.columns:
        df_model_input['property_age'] = 2016 - df_model_input['yearbuilt']
        print("Característica 'property_age' creada y añadida a df_model_input.")
        feature_cols = ['zip_price_rank_median', 'property_age', 'calculatedfinishedsquarefeet', 'bathroomcnt', 'bedroomcnt']
    else:
        print("Advertencia: La columna 'yearbuilt' no se encontró en df_model_input. 'property_age' no pudo ser creada.")
        print("Se utilizarán las features originales incluyendo 'yearbuilt' si está disponible.")
        feature_cols = ['zip_price_rank_median', 'yearbuilt', 'calculatedfinishedsquarefeet', 'bathroomcnt', 'bedroomcnt']

    all_needed_cols = feature_cols + [target_col]
    missing_cols = [col for col in all_needed_cols if col not in df_model_input.columns]

    if missing_cols:
        print(f"Error: Faltan las siguientes columnas necesarias para el modelo KNN: {missing_cols}")
        print(f"Columnas disponibles en df_model_input: {df_model_input.columns.tolist()}")
        df_model_input = None
    else:
        print(f"Features seleccionadas para el modelo: {feature_cols}")
        print(f"Target seleccionado: {target_col}")

        X = df_model_input[feature_cols]
        y = df_model_input[target_col]

        temp_df_for_nan_cleaning = X.copy()
        temp_df_for_nan_cleaning[target_col] = y

        rows_before_nan_drop = len(temp_df_for_nan_cleaning)
        temp_df_for_nan_cleaning.dropna(subset=feature_cols + [target_col], inplace=True)
        rows_after_nan_drop = len(temp_df_for_nan_cleaning)
        print(f"Filas eliminadas por NaNs en features/target seleccionados: {rows_before_nan_drop - rows_after_nan_drop}")

        data_for_knn_analysis = temp_df_for_nan_cleaning

        if data_for_knn_analysis.empty:
            print("No quedan datos después de eliminar NaNs. No se puede continuar con el modelo KNN.")
            df_model_input = None
        else:
            X = data_for_knn_analysis[feature_cols]
            y = data_for_knn_analysis[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Datos divididos: {len(X_train)} entrenamiento ({len(X_train)/len(X)*100:.2f}%), {len(X_test)} prueba ({len(X_test)/len(X)*100:.2f}%).")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            print("Features escaladas.")

            # --- PASO 7: Análisis de `k` óptimo para KNN ---
            print(f"\n--- PASO 7: Analizando k óptimo para KNN Regressor (prediciendo '{target_col}') ---")
            k_range = range(1, 31)
            mse_values = []

            print("Calculando MSE para diferentes valores de k (usando weights='uniform')...")
            for k_val in k_range:
                knn_model = KNeighborsRegressor(n_neighbors=k_val)
                knn_model.fit(X_train_scaled, y_train)
                y_pred = knn_model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                mse_values.append(mse)

            plt.figure(figsize=(12, 6))
            plt.plot(k_range, mse_values, marker='o', linestyle='-')
            plt.title(f'MSE vs. Número de Vecinos (k) para KNN Regressor (Prediciendo {target_col}, weights=uniform)')
            plt.xlabel('Número de Vecinos (k)')
            plt.ylabel('Mean Squared Error (MSE)')
            plt.xticks(list(k_range))
            plt.grid(True)
            plt.show()

            if mse_values:
                optimal_k_mse = min(mse_values)
                optimal_k_value = k_range[mse_values.index(optimal_k_mse)]
                print(f"\nEl MSE mínimo (con weights='uniform') es {optimal_k_mse:.2f} para k = {optimal_k_value}.")
            else:
                print("\nNo se pudieron calcular los valores de MSE.")
            print("Busca en el gráfico un 'codo' o el punto donde el MSE deja de disminuir significativamente.")

            # --- PASO 8: Evaluación detallada con k elegido ---
            print(f"\n--- PASO 8: Evaluación detallada con k elegido (weights='uniform', prediciendo '{target_col}') ---")
            k_elegido = 22

            print(f"Evaluando el modelo con k={k_elegido} y weights='uniform'...")
            knn_model_elegido = KNeighborsRegressor(n_neighbors=k_elegido)
            knn_model_elegido.fit(X_train_scaled, y_train)

            y_pred_elegido = knn_model_elegido.predict(X_test_scaled)

            mse_elegido = mean_squared_error(y_test, y_pred_elegido)
            rmse_elegido = np.sqrt(mse_elegido)
            print(f"  Mean Squared Error (MSE) para k={k_elegido}: {mse_elegido:.2f}")
            print(f"  Root Mean Squared Error (RMSE) para k={k_elegido}: {rmse_elegido:.2f} (error promedio en unidades de '{target_col}')")

            r2_elegido = r2_score(y_test, y_pred_elegido)
            print(f"  Coeficiente de Determinación (R²) para k={k_elegido}: {r2_elegido:.4f}")
            print(f"  Esto significa que el modelo con k={k_elegido} explica aproximadamente el {r2_elegido*100:.2f}% de la variabilidad en '{target_col}'.")

            if np.all(y_test > 0):
                mape_elegido = mean_absolute_percentage_error(y_test, y_pred_elegido)
                print(f"  Error Porcentual Absoluto Medio (MAPE) para k={k_elegido}: {mape_elegido:.4f} ({mape_elegido*100:.2f}%)")
                print(f"  Esto significa que, en promedio, las predicciones del modelo con k={k_elegido} se desvían en un {mape_elegido*100:.2f}% del valor real de '{target_col}'.")
                if mape_elegido < 1:
                     print(f"  Podrías interpretar una 'efectividad porcentual' (basada en MAPE) como {(1-mape_elegido)*100:.2f}%.")
            else:
                print(f"  MAPE no se calculó o podría no ser fiable para k={k_elegido} porque '{target_col}' (y_test) contiene valores cero o negativos.")
                print(f"  Considera R² y RMSE como las principales métricas de rendimiento en este caso.")

            # Gráfico de Valores Reales vs. Predichos (ya existente)
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred_elegido, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
            plt.xlabel(f"Valores Reales ({target_col})")
            plt.ylabel(f"Valores Predichos ({target_col})")
            plt.title(f"Valores Reales vs. Predichos para k={k_elegido} (weights='uniform')")
            plt.grid(True)
            plt.show()

            # <<< INICIO NUEVO CÓDIGO: Gráfico de Residuos >>>
            print("\nGenerando gráfico de residuos...")
            residuals = y_test - y_pred_elegido
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred_elegido, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--') # Línea horizontal en y=0
            plt.xlabel(f"Valores Predichos ({target_col})")
            plt.ylabel("Residuos (Real - Predicho)")
            plt.title(f"Gráfico de Residuos para k={k_elegido} (weights='uniform')")
            plt.grid(True)
            plt.show()
            print("Observa el gráfico de residuos para identificar patrones o heterocedasticidad.")
            # <<< FIN NUEVO CÓDIGO >>>

else:
    if df_model_input is not None and df_model_input.empty :
         print("El DataFrame de entrada para el modelo KNN está vacío.")
    else: # df_model_input es None
         print("No se pudo cargar o generar el DataFrame de entrada para el modelo KNN.")

print("-" * 50)
# --- Resumen Final ---
if df_model_input is not None and not data_for_knn_analysis.empty:
    print("\n✅ Proceso de análisis de k para KNN completado.")
    if 'k_elegido' in locals() or 'optimal_k_value' in locals():
        k_final_eval_paso8 = k_elegido if 'k_elegido' in locals() and 'knn_model_elegido' in locals() else "no evaluado en Paso 8"
        print(f"Se evaluaron métricas adicionales (incluyendo R², weights='uniform') para k={k_final_eval_paso8} en Paso 8.")
        if 'optimal_k_value' in locals():
             print(f"El análisis de k en Paso 7 (con weights='uniform') sugirió k={optimal_k_value} como óptimo.")
    print(f"Se analizó k para predecir '{target_col}' usando las features: {feature_cols if 'feature_cols' in locals() else 'no definidas'}.")
else:
    print("\n❌ Proceso de análisis de k para KNN no pudo completarse debido a problemas con los datos.")
