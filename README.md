# Análisis de supervivencia de los pasajeros del Titanic

Proyecto de análisis exploratorio de datos en Python para la práctica *Git, GitHub y reproducibilidad de proyectos de datos* (asignatura Big Data). Incluye limpieza, preprocesamiento, creación de variables, análisis y visualizaciones. **No se utiliza ningún modelo de Machine Learning.**

## Dataset

- **Nombre:** Titanic - Machine Learning from Disaster (archivo `train.csv`).
- **Fuente:** Kaggle - https://www.kaggle.com/c/titanic/data
- **Descripción:** 891 pasajeros del Titanic con 12 variables (identificador, supervivencia, clase, nombre, sexo, edad, familiares a bordo, billete, tarifa, cabina y puerto de embarque). El archivo ya está incluido en `data/train.csv`, por lo que no hace falta descargarlo aparte.

## Objetivo

Identificar qué características de los pasajeros (sexo, clase, edad, compañía a bordo y tarifa pagada) se asocian con la supervivencia, y hacerlo en un proyecto que cualquier persona pueda reproducir a partir de este repositorio.

## Requisitos

- Python 3.11 o superior (necesario para la versión de pandas fijada).
- Git.
- Las dependencias listadas en `requirements.txt` (pandas y matplotlib con sus dependencias).

## Instalación

```bash
git clone URL_DEL_REPOSITORIO
cd titanic-analisis
python -m venv .venv
```

Activar el entorno virtual:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python src/analysis.py
```

El script imprime la exploración, la limpieza y los análisis en la terminal, y guarda todos los resultados en `outputs/resultados/` (informes de texto, tablas CSV, el dataset limpio y 6 gráficos). Se puede ejecutar desde cualquier carpeta porque las rutas son relativas al proyecto.

## Estructura del repositorio

```
titanic-analisis/
├── data/
│   └── train.csv               # dataset original
├── src/
│   └── analysis.py             # script completo del análisis
├── outputs/
│   └── resultados/             # informes, tablas, dataset limpio y gráficos
├── README.md
├── requirements.txt
└── .gitignore
```

## Análisis realizados

### 1. Exploración inicial
891 pasajeros y 12 columnas. Valores faltantes en `Age` (177; 19.9 %), `Cabin` (687; 77.1 %) y `Embarked` (2; 0.2 %). No hay registros duplicados.

### 2. Tratamiento de valores faltantes (decisiones)

| Variable | Faltantes | Decisión | Justificación |
|---|---|---|---|
| `Age` | 177 (19.9 %) | Imputar con la **mediana por sexo y clase** y conservar la marca `Age_imputada` | La edad difiere entre grupos (p. ej. la 1.ª clase es de mayor edad), así que la mediana por grupo es más realista que una mediana global. La mediana es robusta a valores extremos. |
| `Cabin` | 687 (77.1 %) | **No imputar**: crear `TieneCabina` y `Deck` (`Desconocida` si falta) y eliminar `Cabin` | Con más de tres cuartas partes vacías, imputar inventaría información. El hecho de no tener cabina registrada es en sí mismo informativo. |
| `Embarked` | 2 (0.2 %) | Imputar con la **moda** (`Southampton`) | Solo son 2 filas; la moda es la opción más simple y con impacto mínimo. |

Además se traducen `Sex` y `Embarked` a etiquetas legibles y se añade `Survived_label`.

### 3. Variables nuevas

- `FamilySize = SibSp + Parch + 1`
- `Viaja`: `Solo` (FamilySize = 1) o `Acompañado`.
- `AgeGroup`: **Niño** (0-12), **Joven** (13-25), **Adulto** (26-59), **Adulto mayor** (60 o más).
- `FareGroup`: cuartiles de la tarifa (Q1 barata ... Q4 cara).
- `Title`: título extraído del nombre (Mr, Mrs, Miss, Master, Otro).

### 4. Preguntas respondidas

1. ¿Qué porcentaje de pasajeros sobrevivió?
2. ¿Cómo cambia la supervivencia entre hombres y mujeres?
3. ¿Cómo cambia según la clase del pasajero?
4. ¿Qué grupos de edad presentan mayor supervivencia?
5. ¿Viajar solo o acompañado se relaciona con la supervivencia? (incluye detalle por tamaño de familia)
6. ¿Existe relación entre la tarifa pagada y la supervivencia?
7. Análisis adicional: supervivencia por clase y sexo combinados.
8. Análisis adicional: supervivencia por puerto de embarque.

### 5. Visualizaciones (en `outputs/resultados/`)

1. `grafico_1_supervivencia_global.png`
2. `grafico_2_supervivencia_sexo_y_clase.png`
3. `grafico_3_supervivencia_clase_sexo.png`
4. `grafico_4_supervivencia_edad.png`
5. `grafico_5_supervivencia_compania.png`
6. `grafico_6_supervivencia_tarifa.png`

## Resultados y conclusiones

- **Supervivencia global:** sobrevivieron 342 de 891 pasajeros (**38.4 %**).
- **Sexo:** es el factor con mayor diferencia. Sobrevivió el **74.2 %** de las mujeres frente al **18.9 %** de los hombres.
- **Clase:** la supervivencia baja con la clase: **63.0 %** en 1.ª, **47.3 %** en 2.ª y **24.2 %** en 3.ª.
- **Sexo y clase combinados:** el efecto del sexo se mantiene en todas las clases (mujeres de 1.ª: 96.8 %; hombres de 3.ª: 13.5 %). Las mujeres de 3.ª clase (50.0 %) sobrevivieron más que los hombres de 1.ª clase (36.9 %).
- **Edad:** los **niños (0-12)** tienen la mayor supervivencia (**58.0 %**); los adultos mayores (60+) la menor (**26.9 %**, pero solo 26 pasajeros).
- **Compañía:** quienes viajaron acompañados sobrevivieron más (**50.6 %**) que quienes viajaron solos (**30.4 %**). La relación no es lineal: las familias de 2 a 4 personas tienen las mejores tasas (55-72 %), mientras que las de 5 o más caen por debajo de la media.
- **Tarifa:** hay una relación positiva (correlación 0.257). Los pasajeros del cuartil de tarifa más alta sobrevivieron un **58.1 %** frente al **19.7 %** del más bajo. Buena parte de este efecto coincide con la clase, ya que la tarifa y la clase están muy relacionadas.
- **Puerto:** los embarcados en Cherbourg tuvieron mayor supervivencia (55.4 %) que los de Southampton (33.9 %), probablemente porque en Cherbourg embarcó una mayor proporción de pasajeros de 1.ª clase.

**Limitaciones:** son asociaciones observadas, no relaciones causales. Además, ~20 % de las edades son estimaciones (imputadas), lo que puede afectar a los resultados por grupo de edad, y algunos grupos son pequeños (adultos mayores, familias grandes).

## Reproducibilidad

El proyecto se puede reproducir usando únicamente la URL del repositorio, Git, Python, este README y `requirements.txt`: el dataset y todas las dependencias fijadas están incluidos en el repositorio.
