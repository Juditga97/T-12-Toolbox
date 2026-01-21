import pandas as pd


def tipifica_variables(df):
    """
    Tipifica las variables de un DataFrame en función de su tipo y cardinalidad.

    Argumentos:
    df (pandas.DataFrame): DataFrame cuyas variables se desean analizar y tipificar.

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



