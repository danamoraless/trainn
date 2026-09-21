"""Análisis exploratorio del dataset Titanic (train.csv).

Uso:  python src/analysis.py
Todas las rutas son relativas a la raíz del proyecto, por lo que el script
funciona sin importar desde qué carpeta se ejecute.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: permite ejecutar en cualquier entorno
import matplotlib.pyplot as plt
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


def limpieza(df):
    """Trata los valores faltantes y devuelve un DataFrame limpio.

    Decisiones (documentadas también en el README):
    - Age (19.9 % faltante): se imputa con la mediana de la edad según Sex y
      Pclass, porque la edad varía entre esos grupos. Se conserva la columna
      Age_imputada para saber qué valores son estimados.
    - Cabin (77.1 % faltante): demasiados faltantes para imputar. Se crea
      TieneCabina (1/0) y se extrae la cubierta (Deck; 'Desconocida' si falta).
      Un dato faltante aquí es informativo: suele indicar pasajeros sin cabina
      registrada (mayoría de 3.ª clase).
    - Embarked (0.2 % faltante, 2 filas): se imputa con la moda ('S').
    """
    df = df.copy()
    log = []

    # Age
    n_age = int(df["Age"].isna().sum())
    df["Age_imputada"] = df["Age"].isna().astype(int)
    mediana = df.groupby(["Sex", "Pclass"])["Age"].transform("median")
    df["Age"] = df["Age"].fillna(mediana)
    log.append(f"Age: {n_age} valores imputados con la mediana por Sex y Pclass.")

    # Cabin
    n_cabin = int(df["Cabin"].isna().sum())
    df["TieneCabina"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("Desconocida")
    df = df.drop(columns=["Cabin"])
    log.append(
        f"Cabin: {n_cabin} faltantes; se crea TieneCabina y Deck y se elimina la columna original."
    )

    # Embarked
    moda = df["Embarked"].mode()[0]
    n_emb = int(df["Embarked"].isna().sum())
    df["Embarked"] = df["Embarked"].fillna(moda)
    log.append(f"Embarked: {n_emb} valores imputados con la moda ('{moda}').")

    # Transformaciones de tipo / etiquetas legibles
    df["Sex"] = df["Sex"].map({"male": "hombre", "female": "mujer"})
    df["Survived_label"] = df["Survived"].map({0: "No sobrevivió", 1: "Sobrevivió"})
    df["Embarked"] = df["Embarked"].map(
        {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    )
    log.append("Transformaciones: Sex y Embarked con etiquetas legibles; se añade Survived_label.")

    restantes = int(df.drop(columns=[]).isna().sum().sum())
    log.append(f"Valores faltantes tras la limpieza: {restantes}")
    texto = "=== LIMPIEZA Y PREPROCESAMIENTO ===\n" + "\n".join(log)
    print("\n" + texto)
    (OUT_DIR / "02_limpieza.txt").write_text(texto, encoding="utf-8")
    df.to_csv(OUT_DIR / "titanic_limpio.csv", index=False)
    return df


def nuevas_variables(df):
    """Crea variables derivadas (todas documentadas en el README).

    - FamilySize = SibSp + Parch + 1
    - Viaja: 'Solo' si FamilySize == 1, 'Acompañado' en caso contrario
    - AgeGroup: Niño (0-12), Joven (13-25), Adulto (26-59), Adulto mayor (60+)
    - FareGroup: cuartiles de la tarifa (Q1 más barata ... Q4 más cara)
    - Title: título extraído del nombre, agrupado en Mr / Mrs / Miss / Master / Otro
    """
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Viaja"] = df["FamilySize"].apply(lambda n: "Solo" if n == 1 else "Acompañado")
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 25, 59, 120],
        labels=["Niño", "Joven", "Adulto", "Adulto mayor"],
    )
    df["FareGroup"] = pd.qcut(df["Fare"], 4, labels=["Q1 (barata)", "Q2", "Q3", "Q4 (cara)"])
    titulo = df["Name"].str.extract(r",\s*([^.]+)\.")[0].str.strip()
    df["Title"] = titulo.where(titulo.isin(["Mr", "Mrs", "Miss", "Master"]), "Otro")
    print("\n=== NUEVAS VARIABLES ===")
    print("Creadas: FamilySize, Viaja, AgeGroup, FareGroup, Title")
    return df


def tasa(df, col):
    """Tabla con nº de pasajeros y % de supervivencia por categoría de `col`."""
    t = df.groupby(col, observed=True).agg(
        pasajeros=("Survived", "size"),
        sobrevivientes=("Survived", "sum"),
        supervivencia_pct=("Survived", lambda x: round(x.mean() * 100, 1)),
    )
    return t


def analisis(df):
    """Ejecuta los análisis y guarda los resultados en outputs/resultados."""
    partes = []
    tablas = {}

    total = len(df)
    vivos = int(df["Survived"].sum())
    partes.append(
        "1) Supervivencia global\n"
        f"   {vivos} de {total} pasajeros sobrevivieron ({vivos / total * 100:.1f} %)."
    )

    tablas["supervivencia_por_sexo"] = tasa(df, "Sex")
    partes.append("2) Supervivencia por sexo\n" + tablas["supervivencia_por_sexo"].to_string())

    tablas["supervivencia_por_clase"] = tasa(df, "Pclass")
    partes.append("3) Supervivencia por clase\n" + tablas["supervivencia_por_clase"].to_string())

    tablas["supervivencia_por_edad"] = tasa(df, "AgeGroup")
    partes.append("4) Supervivencia por grupo de edad\n" + tablas["supervivencia_por_edad"].to_string())

    tablas["supervivencia_por_compania"] = tasa(df, "Viaja")
    partes.append("5) Viajar solo vs acompañado\n" + tablas["supervivencia_por_compania"].to_string())

    tablas["supervivencia_por_tamano_familia"] = tasa(df, "FamilySize")
    partes.append(
        "   Detalle por tamaño de familia\n" + tablas["supervivencia_por_tamano_familia"].to_string()
    )

    tablas["supervivencia_por_tarifa"] = tasa(df, "FareGroup")
    tarifa = df.groupby("Survived_label")["Fare"].agg(["mean", "median"]).round(2)
    partes.append(
        "6) Tarifa y supervivencia\n"
        + tablas["supervivencia_por_tarifa"].to_string()
        + "\n   Tarifa media/mediana según supervivencia:\n"
        + tarifa.to_string()
        + f"\n   Correlación Fare-Survived: {df['Fare'].corr(df['Survived']):.3f}"
    )

    cruce = (
        df.pivot_table(index="Pclass", columns="Sex", values="Survived", aggfunc="mean") * 100
    ).round(1)
    tablas["supervivencia_sexo_clase"] = cruce
    partes.append("7) Supervivencia (%) por clase y sexo\n" + cruce.to_string())

    tablas["supervivencia_por_puerto"] = tasa(df, "Embarked")
    partes.append("8) Supervivencia por puerto de embarque\n" + tablas["supervivencia_por_puerto"].to_string())

    texto = "=== ANÁLISIS ===\n" + "\n\n".join(partes)
    print("\n" + texto)
    (OUT_DIR / "03_analisis.txt").write_text(texto, encoding="utf-8")
    for nombre, t in tablas.items():
        t.to_csv(OUT_DIR / f"tabla_{nombre}.csv")
    return tablas


def _barras_pct(ax, serie, titulo, xlabel, colores=None):
    """Dibuja barras de % de supervivencia con etiqueta encima."""
    barras = ax.bar(serie.index.astype(str), serie.values, color=colores or "#4C72B0")
    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Supervivencia (%)")
    ax.set_ylim(0, 100)
    ax.axhline(38.4, color="gray", linestyle="--", linewidth=1)
    ax.text(len(serie) - 0.5, 40, "media global 38.4 %", ha="right", fontsize=8, color="gray")
    for b, v in zip(barras, serie.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)


def visualizaciones(df, tablas):
    """Genera y guarda las visualizaciones en outputs/resultados."""
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

    # 1. Supervivencia global
    fig, ax = plt.subplots(figsize=(5, 5))
    conteo = df["Survived_label"].value_counts().reindex(["Sobrevivió", "No sobrevivió"])
    ax.pie(conteo, labels=conteo.index, autopct="%1.1f%%", startangle=90,
           colors=["#55A868", "#C44E52"], wedgeprops={"edgecolor": "white"})
    ax.set_title("Supervivencia global de los pasajeros", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "grafico_1_supervivencia_global.png")
    plt.close(fig)

    # 2. Sexo y clase
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _barras_pct(axes[0], tablas["supervivencia_por_sexo"]["supervivencia_pct"],
                "Supervivencia por sexo", "Sexo", ["#DD8452", "#4C72B0"])
    _barras_pct(axes[1], tablas["supervivencia_por_clase"]["supervivencia_pct"],
                "Supervivencia por clase", "Clase del pasajero")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "grafico_2_supervivencia_sexo_y_clase.png")
    plt.close(fig)

    # 3. Sexo x clase (barras agrupadas)
    cruce = tablas["supervivencia_sexo_clase"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cruce.plot(kind="bar", ax=ax, color=["#DD8452", "#4C72B0"], rot=0)
    ax.set_title("Supervivencia por clase y sexo", fontweight="bold")
    ax.set_xlabel("Clase del pasajero")
    ax.set_ylabel("Supervivencia (%)")
    ax.set_ylim(0, 105)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.1f", fontsize=8, padding=2)
    ax.legend(title="Sexo")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "grafico_3_supervivencia_clase_sexo.png")
    plt.close(fig)

    # 4. Grupos de edad
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _barras_pct(ax, tablas["supervivencia_por_edad"]["supervivencia_pct"],
                "Supervivencia por grupo de edad", "Grupo de edad")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "grafico_4_supervivencia_edad.png")
    plt.close(fig)

    # 5. Compañía y tamaño de familia
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _barras_pct(axes[0], tablas["supervivencia_por_compania"]["supervivencia_pct"],
                "Viajar solo vs acompañado", "Tipo de viaje")
    _barras_pct(axes[1], tablas["supervivencia_por_tamano_familia"]["supervivencia_pct"],
                "Supervivencia por tamaño de familia", "Tamaño de familia (FamilySize)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "grafico_5_supervivencia_compania.png")
    plt.close(fig)

    # 6. Tarifa
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _barras_pct(axes[0], tablas["supervivencia_por_tarifa"]["supervivencia_pct"],
                "Supervivencia por cuartil de tarifa", "Cuartil de tarifa")
    for etiqueta, color in [("No sobrevivió", "#C44E52"), ("Sobrevivió", "#55A868")]:
        axes[1].hist(df.loc[df["Survived_label"] == etiqueta, "Fare"].clip(upper=150),
                     bins=30, alpha=0.6, label=etiqueta, color=color)
    axes[1].set_title("Distribución de la tarifa (recortada en 150)", fontweight="bold")
    axes[1].set_xlabel("Tarifa")
    axes[1].set_ylabel("Pasajeros")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "grafico_6_supervivencia_tarifa.png")
    plt.close(fig)

    print("\n=== VISUALIZACIONES ===\nSe guardaron 6 gráficos en outputs/resultados/")


def main():
    df = cargar_datos()
    exploracion_inicial(df)
    df = limpieza(df)
    df = nuevas_variables(df)
    df.to_csv(OUT_DIR / "titanic_limpio.csv", index=False)
    tablas = analisis(df)
    visualizaciones(df, tablas)


if __name__ == "__main__":
    main()
