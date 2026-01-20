import pandas as pd

def tipifica_variables(df):
    """
    Tipifica las variables de un DataFrame según su tipo y cardinalidad.

    Argumentos:
    df (pandas.DataFrame): DataFrame cuyas variables se desean analizar y tipificar.

    Retorna:
    pandas.DataFrame: DataFrame resumen que contiene, para cada variable,
    su tipo de dato, número de valores únicos, proporción de valores únicos
    y la tipificación asignada.
    """

    resumen = []
    n_filas = df.shape[0]

    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        prop_unique = n_unique / n_filas

        # Variables numéricas
        if dtype in ["int64", "float64"]:
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

