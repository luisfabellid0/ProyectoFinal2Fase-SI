import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Configuraciones de visualización
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

print("Librerías importadas.")

# --- Carga del Dataset ---
# ASUNCIÓN CRÍTICA: Se asume que 'train_2016_v2.csv' contiene todas las columnas mencionadas,
# incluyendo características de propiedades.
train_file_path = 'train_2016_v2.csv' # Asegúrate que este archivo esté subido a Colab

try:
    print(f"Cargando '{train_file_path}'...")
    # Especificar tipos de datos para optimizar memoria si es necesario,
    # pero por ahora cargaremos directamente para inspeccionar.
    # Si tienes problemas de memoria, puedes añadir un diccionario dtype aquí.
    df = pd.read_csv(train_file_path, parse_dates=['transactiondate'])
    print(f"'{train_file_path}' cargado. Forma inicial: {df.shape}")

    print("\n--- Primeras 5 filas del dataset ---")
    print(df.head())

    print("\n--- Información del DataFrame (columnas y tipos de datos) ---")
    df.info(verbose=True, show_counts=True)

    print("\n--- Conteo de Valores Faltantes (NaNs) por Columna (Top 20) ---")
    print(df.isnull().sum().sort_values(ascending=False).head(20))

except FileNotFoundError:
    print(f"Error: No se encontró el archivo {train_file_path}.")
    print("Asegúrate de que el archivo esté en el directorio correcto de Colab.")
    df = None
except Exception as e:
    print(f"Ocurrió un error al cargar el archivo: {e}")
    df = None


if df is not None and not df.empty:
    print("\n\n--- PARTE 3: PREPROCESAMIENTO Y FEATURE ENGINEERING ---")

    # 3.1 Feature Engineering a partir de 'transactiondate'
    if 'transactiondate' in df.columns:
        print("\nCreando características a partir de 'transactiondate'...")
        df['transaction_year'] = df['transactiondate'].dt.year
        df['transaction_month'] = df['transactiondate'].dt.month
        df['transaction_dayofweek'] = df['transactiondate'].dt.dayofweek
        print("Características de fecha creadas: 'transaction_year', 'transaction_month', 'transaction_dayofweek'")
    else:
        print("Advertencia: Columna 'transactiondate' no encontrada para feature engineering.")

    # 3.2 Transformar 'logerror' en categorías para clasificación
    if 'logerror' in df.columns:
        print("\nTransformando 'logerror' en categorías...")
        # Umbrales: considera ajustarlos según la distribución y el objetivo del problema.
        # [-inf, bajo, alto, inf] => 3 clases
        # Puedes experimentar con cuantiles para clases más balanceadas si es necesario
        # low_q = df['logerror'].quantile(0.33)
        # high_q = df['logerror'].quantile(0.66)
        # bins = [-float('inf'), low_q, high_q, float('inf')]
        bins = [-float('inf'), -0.04, 0.04, float('inf')] # Umbrales del código original
        labels = ['Subestimacion', 'Prediccion_Precisa', 'Sobreestimacion']
        df['logerror_category'] = pd.cut(df['logerror'], bins=bins, labels=labels, right=True, include_lowest=True)

        # Eliminar filas donde 'logerror' o la categoría resultante sea NaN (si 'logerror' tiene NaNs)
        df.dropna(subset=['logerror', 'logerror_category'], inplace=True)
        print("Distribución de 'logerror_category':")
        print(df['logerror_category'].value_counts(normalize=True) * 100)

        le = LabelEncoder()
        y = le.fit_transform(df['logerror_category'])
        class_names = le.classes_
        print(f"Variable objetivo 'y' codificada. Clases mapeadas: {dict(zip(range(len(class_names)), class_names))}")
    else:
        print("Error: Columna 'logerror' no encontrada. No se puede crear la variable objetivo.")
        y = None # Marcar y como None para evitar errores posteriores

    # 3.3 Selección de Características (X)
    if y is not None:
        print("\nSeleccionando características para X...")
        # Columnas a eliminar de las features:
        # - IDs únicos: parcelid
        # - Target y relacionadas directamente con el target: logerror, logerror_category
        # - Fecha original (ya procesada): transactiondate
        # - Columnas que son precios directos o valores muy directos (para evitar data leakage al predecir 'logerror_category'):
        #   'price', 'taxvaluedollarcnt'
        #   'taxamount' podría ser útil y no tan directo como taxvaluedollarcnt.
        #   'zip_price_rank' podría ser una característica derivada útil.

        potential_features = [
            'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'yearbuilt',
            'taxamount', 'latitude', 'longitude', 'regionidzip', 'regionidcounty',
            'zip_price_rank', # Asumiendo que esta no es un proxy directo del precio que queremos evitar
            'transaction_year', 'transaction_month', 'transaction_dayofweek' # Nuevas features de fecha
        ]

        # Mantener solo las columnas que existen en el DataFrame
        features_to_use = [col for col in potential_features if col in df.columns]

        # Columnas a dropear explícitamente si existen y no están en features_to_use
        cols_to_drop_explicitly = ['parcelid', 'logerror', 'logerror_category', 'transactiondate', 'price', 'taxvaluedollarcnt']

        # X contendrá solo las features_to_use
        X = df[features_to_use].copy()

        print(f"Características (X) seleccionadas: {X.columns.tolist()}")
        print(f"Dimensiones de X (antes de imputar NaNs): {X.shape}")
        print(f"Dimensiones de y: {y.shape if y is not None else 'No definido'}")

        # 3.4 Manejo de Valores Faltantes en X
        print("\nManejando valores faltantes en X...")
        missing_percentage = (X.isnull().sum() / len(X)) * 100
        print("Porcentaje de NaNs por columna en X (antes de imputar):")
        print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))

        # Umbral para eliminar columnas con demasiados NaNs (ej. > 70%)
        threshold_high_nan_percentage = 70
        cols_to_drop_due_to_nan = missing_percentage[missing_percentage > threshold_high_nan_percentage].index.tolist()

        if cols_to_drop_due_to_nan:
            print(f"\nEliminando columnas con más del {threshold_high_nan_percentage}% de NaNs: {cols_to_drop_due_to_nan}")
            X.drop(columns=cols_to_drop_due_to_nan, inplace=True)
            print("Dimensiones de X después de eliminar columnas con muchos NaNs:", X.shape)
        else:
            print(f"\nNo hay columnas con más del {threshold_high_nan_percentage}% de NaNs para eliminar.")

        # Imputación de Valores Faltantes restantes (solo numéricas por ahora)
        numeric_cols_in_X = X.select_dtypes(include=np.number).columns
        if not X[numeric_cols_in_X].empty:
            print(f"\nImputando NaNs en {len(numeric_cols_in_X)} columnas numéricas de X con la mediana...")
            imputer_numeric = SimpleImputer(strategy='median')
            X[numeric_cols_in_X] = imputer_numeric.fit_transform(X[numeric_cols_in_X])
            print("Imputación de NaNs en columnas numéricas completada.")
        else:
            print("\nNo hay columnas numéricas en X para imputar o X está vacío.")

        print("\nTotal de valores faltantes en X después de la imputación:", X.isnull().sum().sum())
        if X.isnull().sum().sum() > 0:
             print("¡ADVERTENCIA! Todavía hay NaNs en X. Revisa las columnas no numéricas o el proceso de imputación.")
    else:
        X = None # Marcar X como None
        print("No se pudo definir X porque y no está disponible.")
else:
    print("El DataFrame 'df' no está definido o está vacío. Ejecuta la celda anterior para cargar los datos.")
    X, y = None, None
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint # Para especificar rangos en RandomizedSearchCV

if X is not None and y is not None and not X.empty and len(y) == X.shape[0]:
    print("\n\n--- PARTE 4: DIVISIÓN DE DATOS Y ENTRENAMIENTO DEL MODELO (OPTIMIZADO) ---")

    # 4.1 División en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print("\nDatos divididos en conjuntos de entrenamiento y prueba:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    feature_names_for_model = X_train.columns.tolist()

    # 4.2 (Opcional) Submuestreo para acelerar la búsqueda de hiperparámetros
    # Si el entrenamiento sigue siendo muy lento, considera usar una fracción de los datos para la búsqueda.
    # Por ejemplo, para usar el 50% de los datos de entrenamiento para la búsqueda:
    # X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, train_size=0.5, random_state=42, stratify=y_train)
    # print(f"\nUsando una muestra para RandomizedSearchCV: X_train_sample: {X_train_sample.shape}")
    # Y luego usar X_train_sample, y_train_sample en random_search.fit()

    # Usaremos el X_train completo por defecto. Descomenta las líneas de arriba para usar una muestra.
    X_tune = X_train
    y_tune = y_train

    # 4.3 Optimización de Hiperparámetros con RandomizedSearchCV
    print("\nConfigurando RandomizedSearchCV para DecisionTreeClassifier...")

    # Espacio de parámetros para RandomizedSearchCV
    # Puedes usar distribuciones (ej. randint) para un muestreo más aleatorio si lo deseas
    param_dist = {
        'max_depth': [8, 10, 12, 15, 20, None], # None significa sin límite
        'min_samples_split': randint(50, 400), # Enteros aleatorios entre 50 y 400
        'min_samples_leaf': randint(25, 200),  # Enteros aleatorios entre 25 y 200
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', None],
        'ccp_alpha': [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01] # Cost Complexity Pruning
    }

    # Número de combinaciones de parámetros a probar.
    # Ajusta n_iter según el tiempo disponible. Un valor entre 30-100 suele ser un buen compromiso.
    # Para una ejecución más rápida inicial, puedes probar con n_iter = 20 o 30.
    # Para una búsqueda más exhaustiva (si tienes tiempo), incrementa a 50 o 100.
    n_iter_search = 50 # Reducido de las 432 combinaciones de GridSearchCV

    random_search = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=n_iter_search, # Número de iteraciones (combinaciones a probar)
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        random_state=42, # Para reproducibilidad de la búsqueda aleatoria
        verbose=1
    )

    print(f"Entrenando con RandomizedSearchCV (probando {n_iter_search} combinaciones, esto puede tardar)...")
    random_search.fit(X_tune, y_tune) # Usa X_tune, y_tune (que puede ser el completo o la muestra)

    print("\nMejores hiperparámetros encontrados por RandomizedSearchCV:")
    print(random_search.best_params_)

    # Usar el mejor estimador encontrado
    best_dt_classifier = random_search.best_estimator_

    # Opcional: Si usaste una muestra para la búsqueda, ahora puedes re-entrenar el mejor modelo con el X_train completo
    # if X_tune.shape[0] < X_train.shape[0]:
    #     print("\nRe-entrenando el mejor modelo encontrado en el conjunto de entrenamiento completo...")
    #     best_dt_classifier.fit(X_train, y_train)
    #     print("Modelo re-entrenado en datos completos.")

    print("\nModelo Decision Tree Classifier entrenado con los mejores hiperparámetros.")

else:
    print("X o y no están definidos, X está vacío, o sus dimensiones no coinciden. Revisa los pasos anteriores.")
    best_dt_classifier = None
    # Asegurar que estas variables estén definidas para que la celda de evaluación no falle si esta celda no se ejecuta bien.
    X_test, y_test, feature_names_for_model, class_names = None, None, None, None
    if 'le' in globals() and hasattr(le, 'classes_'): # Si LabelEncoder 'le' fue ajustado previamente
        class_names = le.classes_
if best_dt_classifier is not None and X_test is not None and y_test is not None:
    print("\n\n--- PARTE 5: EVALUACIÓN DEL MODELO ---")

    y_pred = best_dt_classifier.predict(X_test)
    print("\nPredicciones realizadas sobre el conjunto de prueba.")

    # 5.1 Matriz de Confusión
    print("\n--- Matriz de Confusión ---")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.show()

    # 5.2 Métricas de Clasificación (incluyendo precisión)
    print("\n--- Métricas de Clasificación ---")
    accuracy = accuracy_score(y_test, y_pred)
    # Usamos average='weighted' para tener en cuenta el desbalanceo de clases en las métricas promedio
    precision_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision_w:.4f}") # Precisión ponderada
    print(f"Recall (Weighted): {recall_w:.4f}")
    print(f"F1-Score (Weighted): {f1_w:.4f}")

    print("\n--- Reporte de Clasificación Detallado por Clase ---")
    # Muestra precisión, recall, f1-score para cada clase
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # 5.3 Importancia de Características
    if hasattr(best_dt_classifier, 'feature_importances_') and feature_names_for_model is not None:
        print("\n--- Importancia de Características ---")
        importances = best_dt_classifier.feature_importances_

        if len(importances) == len(feature_names_for_model):
            feature_importance_df = pd.DataFrame({
                'Característica': feature_names_for_model,
                'Importancia': importances
            }).sort_values(by='Importancia', ascending=False)

            print("\nTop 10 características más importantes:")
            print(feature_importance_df.head(10))

            plt.figure(figsize=(10, min(10, len(feature_importance_df)) * 0.6))
            sns.barplot(x='Importancia', y='Característica', data=feature_importance_df.head(10), palette="viridis_r")
            plt.title('Top 10 Características Más Importantes')
            plt.xlabel('Puntuación de Importancia')
            plt.ylabel('Características')
            plt.tight_layout()
            plt.show()
        else:
            print("Error: El número de importancias no coincide con el número de nombres de características.")
            print(f"Importancias: {len(importances)}, Nombres de características: {len(feature_names_for_model)}")
    else:
        print("El atributo 'feature_importances_' no está disponible o los nombres de características no se guardaron.")
else:
    print("El modelo (best_dt_classifier) o los datos de prueba no están definidos. Ejecuta las celdas anteriores.")

print("\n\n--- Fin del Script ---")
from sklearn.tree import plot_tree

if 'best_dt_classifier' in globals() and best_dt_classifier is not None:
    if 'feature_names_for_model' in globals() and feature_names_for_model is not None and \
       'class_names' in globals() and class_names is not None:

        print("\n\n--- PARTE 6: VISUALIZACIÓN DEL ÁRBOL DE DECISIÓN ---")

        # Determinar la profundidad real del árbol entrenado
        actual_tree_depth = best_dt_classifier.get_depth()
        print(f"Profundidad real del árbol entrenado: {actual_tree_depth}")

        # Definir cuántos niveles quieres visualizar
        # Si el árbol es menos profundo que esto, se mostrará completo.
        levels_to_show = 3
        if actual_tree_depth == 0: # Árbol es solo un nodo raíz (stump)
            print("El árbol es un stump (solo un nodo raíz), mostrando el nodo raíz.")
            levels_to_show = 0
        elif actual_tree_depth < levels_to_show:
            print(f"El árbol tiene menos de {levels_to_show} niveles, se mostrará completo ({actual_tree_depth} niveles).")
            levels_to_show = actual_tree_depth


        plt.figure(figsize=(25, 15)) # Ajusta el tamaño según sea necesario
        plot_tree(
            best_dt_classifier,
            feature_names=feature_names_for_model, # Nombres de tus características
            class_names=class_names,             # Nombres de tus clases objetivo
            filled=True,                         # Colorea los nodos para indicar la clase mayoritaria
            rounded=True,                        # Nodos con esquinas redondeadas
            fontsize=10,                         # Tamaño de la fuente
            max_depth=levels_to_show,            # Muestra solo los primeros 'levels_to_show' niveles
            impurity=True,                       # Muestra la impureza (gini/entropy)
            proportion=False,                    # Muestra conteos de muestras en lugar de proporciones
            label='all'                          # Muestra información en todos los nodos
        )
        plt.title(f"Árbol de Decisión (Primeros {levels_to_show if levels_to_show > 0 else '1'} Nivel(es))", fontsize=20)
        plt.show()

        if levels_to_show < actual_tree_depth and actual_tree_depth > 0 :
            print(f"\nNota: Solo se muestran los primeros {levels_to_show} niveles del árbol.")
            print(f"Para ver más niveles, incrementa la variable 'levels_to_show' y re-ejecuta esta celda.")
            print(f"La profundidad total del árbol es {actual_tree_depth}.")

    else:
        print("No se puede visualizar el árbol: faltan nombres de características o clases.")
        if 'feature_names_for_model' not in globals() or feature_names_for_model is None:
            print("- 'feature_names_for_model' no está definido.")
        if 'class_names' not in globals() or class_names is None:
             print("- 'class_names' no está definido. Asegúrate que el LabelEncoder 'le' se haya ajustado.")

else:
    print("No se puede visualizar el árbol: el modelo 'best_dt_classifier' no ha sido entrenado o no está disponible.")
