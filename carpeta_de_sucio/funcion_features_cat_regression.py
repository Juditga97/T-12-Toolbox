import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def get_features_cat_regression(df: pd.DataFrame, target_col: str, pvalue: float = 0.05):
    """
    Devuelve las columnas categóricas cuya relación con una target numérica es significativa.
    (t-test si 2 grupos, ANOVA si 3+).

    Argumentos:
    df (pd.DataFrame): Datos.
    target_col (str): Columna numérica target.
    pvalue (float): Nivel de significación.

    Retorna:
    list: Lista de columnas categóricas significativas. Si hay error, devuelve [].
    """
    if not isinstance(df, pd.DataFrame) or target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error en df o target_col")
        return []

    cat_cols = [c for c in df.columns if c != target_col and not pd.api.types.is_numeric_dtype(df[c])]
    sig = []

    for c in cat_cols:
        tmp = df[[c, target_col]].dropna()
        groups = [tmp.loc[tmp[c] == v, target_col] for v in tmp[c].unique()]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue

        p = stats.ttest_ind(groups[0], groups[1], equal_var=False).pvalue if len(groups) == 2 else stats.f_oneway(*groups).pvalue
        if p < pvalue:
            sig.append(c)

    return sig

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
    if not isinstance(df, pd.DataFrame) or target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error en df o target_col")
        return []

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