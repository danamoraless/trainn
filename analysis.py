"""Análisis exploratorio del dataset Titanic (train.csv).

Uso:  python src/analysis.py
Todas las rutas son relativas a la raíz del proyecto, por lo que el script
funciona sin importar desde qué carpeta se ejecute.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
OUT_DIR = ROOT / "outputs" / "resultados"


def cargar_datos(path=DATA_PATH):
    """Carga train.csv en un DataFrame."""
    return pd.read_csv(path)


def exploracion_inicial(df):
    """Imprime y guarda la exploración inicial del dataset."""
    lineas = []
    lineas.append("=== EXPLORACIÓN INICIAL ===")
    lineas.append(f"Número de pasajeros (filas): {df.shape[0]}")
    lineas.append(f"Número de columnas: {df.shape[1]}")
    lineas.append(f"Variables: {', '.join(df.columns)}")
    lineas.append("\nTipos de datos:\n" + df.dtypes.to_string())
    faltantes = pd.DataFrame(
        {"faltantes": df.isna().sum(), "porcentaje": (df.isna().mean() * 100).round(2)}
    )
    lineas.append("\nValores faltantes:\n" + faltantes.to_string())
    lineas.append(f"\nRegistros duplicados: {df.duplicated().sum()}")
    lineas.append("\nEstadísticas descriptivas (numéricas):\n" + df.describe().round(2).to_string())
    lineas.append(
        "\nEstadísticas descriptivas (categóricas):\n"
        + df.describe(include=["object", "string"]).to_string()
    )
    texto = "\n".join(lineas)
    print(texto)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "01_exploracion_inicial.txt").write_text(texto, encoding="utf-8")


def main():
    df = cargar_datos()
    exploracion_inicial(df)


if __name__ == "__main__":
    main()
