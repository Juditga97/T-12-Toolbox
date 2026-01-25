import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import stats
from scipy.stats import chi2_contingency


'''Primera Herramienta: Describe_df'''


def describe_df(df):

    """
    Analiza el tipo de datos de la columna, número de missing values, el número de valores únicos y la cardinalidad de un data frame.

    Argumentos:
    df (pd.DataFrame): DataFrame cuyas columnas se desean analizar.
    
    Devuelve:
    DataFrame: con tantas columnas como columnas tenga el df original y con 4 filas, una con cada valor.
        DATA_TYPE: tipo de dato de la columna (object, int, float, bool)
        MISSINGS(%): porcentaje de valores perdidos (NaN) de la columna
        UNIQUE_VALUES: número de valores únicos de la columna
        CARDIN(%): porcentaje de cardinalidad, (valores únicos / valores totales) de la columna
    """

    dict = {}
    list_valores = ["DATA_TYPE","MISSINGS(%)","UNIQUE_VALUES","CARDIN(%)"]
    indices = df.dtypes.index

    DATA_TYPE= df.dtypes
    MISSINGS = df.isna().sum().values/len(df)
    UNIQUE_VALUES = []
    for indice in indices:
        UNIQUE_VALUES.append(len(df[indice].value_counts())/len(df))
    CARDIN = UNIQUE_VALUES

    lista_datos = [DATA_TYPE,MISSINGS,UNIQUE_VALUES,CARDIN]
    for indice, valor in enumerate(list_valores):
        dict[valor] = lista_datos[indice]
    describe = pd.DataFrame(dict).T
    return describe



'''Segunda Herramienta: tipifica_variables'''

def tipifica_variables(df):
    """
    Tipifica las variables de un DataFrame en función de su tipo y cardinalidad.

    Argumentos:
    df (pandas.DataFrame): DataFrame cuyas columnas se desean analizar y tipificar.

    Retorna:
    pandas.DataFrame: DataFrame resumen que contiene, para cada variable:
        - variable (str): nombre de la columna
        - dtype (str): tipo de dato original
        - n_unique (int): número de valores únicos
        - prop_unique (float): proporción de valores únicos (0 a 1)
        - tipo_variable (str): clasificación asignada
    """

    resumen = []
    n_filas = df.shape[0]

    # Caso especial: DataFrame vacío
    if n_filas == 0:
        return pd.DataFrame(
            columns=["variable", "dtype", "n_unique", "prop_unique", "tipo_variable"]
        )

    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        prop_unique = n_unique / n_filas

        # Variables temporales
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            tipo = "Temporal"

        # Variables numéricas
        elif pd.api.types.is_numeric_dtype(df[col]):
            if n_unique == 2:
                tipo = "Numérica binaria"
            elif n_unique <= 15:
                tipo = "Numérica discreta"
            else:
                tipo = "Numérica continua"

        # Variables categóricas
        else:
            if n_unique == 2:
                tipo = "Categórica binaria"
            elif n_unique <= 10:
                tipo = "Categórica baja"
            elif n_unique <= 20:
                tipo = "Categórica media"
            else:
                tipo = "Categórica alta"

        resumen.append({
            "variable": col,
            "dtype": str(dtype),
            "n_unique": n_unique,
            "prop_unique": round(prop_unique, 3),
            "tipo_variable": tipo
        })

    return pd.DataFrame(resumen)



'''Tercera Herramienta: get_features_num_regression'''

def get_features_num_regression (dataframe,
                                 target_col,
                                 umbral_corr = 0.5,
                                 pvalue = None):
    '''
    Selecciona las variables numéricas de un DataFrame que presentan una correlación 
    significativa con una variable objetivo, pensada para un problema de regresión.

    Argumentos:
        dataframe (pd.DataFrame): DataFrame que contiene los datos
        target_col (str): Nombre de la columna objetivo. Debe ser una variable numérica continua o discreta con una cardinalidad alta.
        umbral_corr (float): Umbral mínimo de correlación que debe existir entre una variable y target_col para ser seleccionada. 
        Debe estar entre 0 y 1. Por defecto indicamos 0.5.
        pvalue (float o None): Nivel de significación estadística para el test de hipótesis de la correlación.

    Devuelve:
        list o None: Lista con los nombres de las columnas numéricas que cumplen los criterios establecidos.
        Devuelve None si los parámetros de entrada no son válidos.
    '''

    if not isinstance (dataframe, pd.DataFrame):
        print ("Error: dataframe no es un DataFrame.")
        return None
    
    if target_col not in dataframe.columns:
        print ("Error: target_col no existe en el DataFrame")
        return None
    
    if not pd.api.types.is_numeric_dtype(dataframe[target_col]):
        print ("Error: target_col debe ser una variable numérica")
        return None
    
    if dataframe[target_col].nunique()<10:
        print ("Error: target_col no tiene suficiente cardinalidad para regresión")
        return None
    
    if not isinstance (umbral_corr, float) or not (0<= umbral_corr <=1):
        print ("Error: umbral_corr debe ser un float entre 0 y 1.")
        return None
    
    if pvalue is not None:
        if not isinstance (pvalue, float) or not (0 <=pvalue <=1):
            print ("Error: pvalue debe ser None o un float entre 0 y 1.")
            return None
        
    numeric_cols = dataframe.select_dtypes(include=np.number).columns
    numeric_cols = numeric_cols.drop(target_col)

    selected_features = []

    for col in numeric_cols:
        aux_df = dataframe[[target_col, col]].dropna()

        if len (aux_df) <2:
            continue

        corr, p_val = pearsonr (aux_df[target_col], aux_df[col])

        if abs (corr) >= umbral_corr:
            if pvalue is None or p_val <= pvalue:
                selected_features.append(col)

    return selected_features



'''Cuarta Herramienta: plot_features_numbers_num_regression'''

def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Esta función filtra variables numéricas de un DataFrame basándose en la correlación 
    con una columna objetivo y un umbral de p-value, y genera pairplots.
    """
    
    # --- 1. Chequeo de valores de entrada ---
    if target_col not in df.columns:
        print(f"Error: La columna target_col '{target_col}' no se encuentra en el DataFrame.")
        return []

    # Si la lista columns está vacía, usar todas las columnas numéricas excepto target_col
    if not columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [col for col in numeric_cols if col != target_col]

    valid_columns = []
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            valid_columns.append(col)
        else:
            print(f"Advertencia: La columna '{col}' no es válida o no es numérica y será ignorada.")
    
    if not valid_columns:
        print("No hay columnas numéricas válidas para analizar.")
        return []
    
    # --- 2. Filtrado de columnas basado en correlación y p-value ---
    selected_features = []
    for col in valid_columns:
        # Calcular correlación de Pearson y p-value
        temp_df = df[[target_col, col]].dropna()
        if len(temp_df) < 2:
            continue
            
        r, p = pearsonr(temp_df[target_col], temp_df[col])
        
        # Verificar condiciones de umbral de correlación y p-value
        corr_condition = abs(r) > umbral_corr
        # El nivel de significación es 1-pvalue, por lo que p debe ser < (1 - (1-alpha)) = alpha, si pvalue es alpha
        pvalue_condition = True if pvalue is None else p < pvalue 

        if corr_condition and pvalue_condition:
            selected_features.append(col)

    if not selected_features:
        print(f"No se encontraron columnas que cumplan los criterios de correlación (>{umbral_corr}) y/o p-value.")
        return []

    # --- 3. Visualización (Extra: múltiples pairplots) ---
    features_to_plot = [target_col] + selected_features
    max_cols_per_plot = 4 # 5 incluyendo target_col
    
    if len(selected_features) > max_cols_per_plot:
        print(f"Se encontraron {len(selected_features)} columnas relevantes. Generando múltiples pairplots.")
        for i in range(0, len(selected_features), max_cols_per_plot):
            subset_features = [target_col] + selected_features[i:i + max_cols_per_plot]
            sns.pairplot(df[subset_features])
            plt.show()
    else:
        sns.pairplot(df[features_to_plot])
        plt.show()

    # Devolver los valores de "columns" que cumplen con las condiciones
    return selected_features



'''Quinta Herramienta: get_features_cat_regression'''

def get_features_cat_regression(
    df: pd.DataFrame,
    target_col: str,
    pvalue: float = 0.05
):
    """
    Devuelve las columnas categóricas que tienen relación significativa
    con una target categórica (clasificación) usando Chi-cuadrado.

    Args:
        df (pd.DataFrame): DataFrame de entrada
        target_col (str): Columna target categórica
        pvalue (float): Nivel de significación

    Returns:
        list: Columnas categóricas significativas
    """

    # --- Validaciones ---
    if not isinstance(df, pd.DataFrame):
        print("Error: df no es un DataFrame")
        return []

    if target_col not in df.columns:
        print("Error: target_col no existe")
        return []

    if pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser categórica")
        return []

    cat_cols = [
        c for c in df.columns
        if c != target_col and not pd.api.types.is_numeric_dtype(df[c])
    ]

    significant = []

    for col in cat_cols:
        contingency = pd.crosstab(df[col], df[target_col])

        # Chi-cuadrado necesita al menos una tabla 2x2 válida
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue

        chi2, p, _, _ = chi2_contingency(contingency)

        if p < pvalue:
            significant.append(col)

    return significant



'''Sexta Herramienta: plot_features_cat_regression'''

def plot_features_cat_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list[str] = [],
    pvalue: float = 0.05,
    with_individual_plot: bool = False):
    """
    Plotea histogramas de target_col por categorías para las columnas significativas.

    Argumentos:
    df (pd.DataFrame): Datos.
    target_col (str): Columna numérica target.
    columns (list[str]): Categóricas a evaluar (si [] usa todas).
    pvalue (float): Nivel de significación.
    with_individual_plot (bool): Si True, también plots por categoría.

    Devuelve:
    list: Columnas significativas ploteadas. Si hay error, devuelve [].
    """

    if columns == []:
        columns = [c for c in df.columns if c != target_col and not pd.api.types.is_numeric_dtype(df[c])]
    else:
        columns = [c for c in columns if c in df.columns]

    sig = get_features_cat_regression(df[[*columns, target_col]], target_col, pvalue)
    if sig is None:
        return []

    for c in sig:
        tmp = df[[c, target_col]].dropna()
        levels = tmp[c].unique()

        plt.figure(figsize=(10, 5))
        for v in levels:
            plt.hist(tmp.loc[tmp[c] == v, target_col], bins=20, alpha=0.5, label=str(v))
        plt.title(f"{target_col} por {c}")
        plt.xlabel(target_col)
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.tight_layout()
        plt.show()

        if with_individual_plot:
            for v in levels:
                plt.figure(figsize=(8, 4))
                plt.hist(tmp.loc[tmp[c] == v, target_col], bins=20)
                plt.title(f"{target_col} | {c}={v}")
                plt.xlabel(target_col)
                plt.ylabel("Frecuencia")
                plt.tight_layout()
                plt.show()

    return sig