"""
===============================================================================
 Assignment 13: Data Gathering, Preprocessing & EDA
 Author  : PATAN KHALILULLA KHAN
 Purpose : Pre-ML foundations only - data gathering, cleaning and EDA.
           (No model training in this assignment.)
 Libraries: pandas, numpy, matplotlib, seaborn, requests, sqlite3 ONLY.
===============================================================================
"""

import os
import sqlite3
import json

import numpy as np
import pandas as pd
import matplotlib
# In a Jupyter notebook keep the interactive inline backend; as a plain script
# (headless) use Agg so charts are saved to disk instead.
try:
    get_ipython()  # noqa: F821 - defined inside IPython/Jupyter
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False

if not IN_NOTEBOOK:
    matplotlib.use("Agg")                  # headless backend -> save charts to disk
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# ------------------------------------------------------------------------------
# Global settings
# ------------------------------------------------------------------------------
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid", palette="deep")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHART_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "titanic.csv")          # Kaggle: Titanic dataset
JSON_PATH = os.path.join(DATA_DIR, "products.json")       # small hand-made JSON
DB_PATH = os.path.join(DATA_DIR, "sample.db")             # SQLite database
TMDB_CSV = os.path.join(DATA_DIR, "tmdb_movies.csv")      # Task 4 output
TMDB_SAMPLE = os.path.join(DATA_DIR, "tmdb_sample.json")  # fallback when no API key
TMDB_KEY_FILE = os.path.join(DATA_DIR, "tmdb_api_key.txt")  # optional: put key here


def banner(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def save_chart(fig, filename):
    path = os.path.join(CHART_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart saved] charts/{filename}")


# ==============================================================================
# PART 1 - DATA GATHERING (MULTIPLE SOURCES)
# ==============================================================================

# ------------------------------------------------------------------------------
# TASK 1: Load data from CSV (Kaggle Titanic dataset)
# ------------------------------------------------------------------------------
def task1_load_csv():
    banner("PART 1 - TASK 1: LOAD DATA FROM CSV (Kaggle Titanic dataset)")

    df = pd.read_csv(CSV_PATH)
    print("Shape of the dataset:", df.shape)
    print("\nColumn names:\n", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nBasic info (dtypes + non-null counts):")
    df.info()
    print("\nStatistical summary of numerical columns:")
    print(df.describe())
    return df


# ------------------------------------------------------------------------------
# TASK 2: Load data from JSON and convert to DataFrame
# ------------------------------------------------------------------------------
def task2_load_json():
    banner("PART 1 - TASK 2: LOAD DATA FROM JSON")

    # read_json returns a DataFrame directly when the JSON is a list of records;
    # dtype=... forces price/stock to stay STRINGS (they are fixed in Task 6)
    products = pd.read_json(JSON_PATH, dtype={"price": str, "stock": str})
    print("products.json loaded - shape:", products.shape)
    print(products)
    print("\nColumn dtypes (note: price/stock are strings on purpose, fixed in Task 6):")
    print(products.dtypes)
    return products


# ------------------------------------------------------------------------------
# TASK 3: Load data from a SQL database (SQLite)
# ------------------------------------------------------------------------------
def task3_load_sql():
    banner("PART 1 - TASK 3: LOAD DATA FROM SQL DATABASE (SQLite)")

    # 1) create / connect to sample.db
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 2) create the employees table
    cursor.execute("DROP TABLE IF EXISTS employees")
    cursor.execute("""
        CREATE TABLE employees (
            id         INTEGER PRIMARY KEY,
            name       TEXT NOT NULL,
            department TEXT NOT NULL,
            salary     REAL NOT NULL
        )
    """)

    # 3) insert at least 5 records
    employees = [
        (1, "Patan Khalilulla Khan", "Data Science", 55000.0),
        (2, "Ayesha Siddiqua",       "Engineering",  72000.0),
        (3, "Rahul Verma",           "Engineering",  68000.0),
        (4, "Fatima Noor",           "Human Resources", 48000.0),
        (5, "Vikram Singh",          "Finance",      61000.0),
        (6, "Sana Patel",            "Marketing",    52000.0),
        (7, "Arjun Reddy",           "Data Science", 64000.0),
        (8, "Meena Kumari",          "Engineering",  75000.0),
    ]
    cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?)", employees)
    conn.commit()
    print("Created 'sample.db' with table 'employees' and inserted", len(employees), "records.")

    # 4) read the table into pandas using a SQL query
    query = "SELECT * FROM employees ORDER BY salary DESC"
    df_emp = pd.read_sql_query(query, conn)
    conn.close()

    # 5) display the DataFrame
    print("\nEmployees DataFrame (read via SQL query):")
    print(df_emp)
    print("\nAverage salary per department (SQL aggregation):")
    conn = sqlite3.connect(DB_PATH)
    print(pd.read_sql_query(
        "SELECT department, COUNT(*) AS n_employees, ROUND(AVG(salary), 2) AS avg_salary "
        "FROM employees GROUP BY department ORDER BY avg_salary DESC", conn))
    conn.close()
    return df_emp


# ------------------------------------------------------------------------------
# TASK 4: API mini project - TMDB movie data collector
# ------------------------------------------------------------------------------
def _read_tmdb_key():
    """Look for a TMDB API key in env var TMDB_API_KEY or data/tmdb_api_key.txt."""
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if key:
        return key
    if os.path.exists(TMDB_KEY_FILE):
        with open(TMDB_KEY_FILE, "r", encoding="utf-8") as fh:
            key = fh.read().strip()
        if key:
            return key
    return ""


def _fetch_tmdb(endpoint, api_key):
    url = "https://api.themoviedb.org/3/movie/" + endpoint
    resp = requests.get(url, params={"api_key": api_key, "language": "en-US", "page": 1}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def task4_tmdb_api():
    banner("PART 1 - TASK 4: API MINI PROJECT - TMDB MOVIE DATA COLLECTOR")

    api_key = _read_tmdb_key()
    rows = []

    if api_key:
        # LIVE path: fetch popular + top-rated movies from the TMDB API
        print("TMDB API key found -> fetching LIVE data from api.themoviedb.org ...")
        for endpoint, source in (("popular", "popular"), ("top_rated", "top_rated")):
            payload = _fetch_tmdb(endpoint, api_key)
            for m in payload.get("results", []):
                rows.append({
                    "title": m.get("title"),
                    "release_date": m.get("release_date"),
                    "rating": m.get("vote_average"),
                    "popularity": m.get("popularity"),
                    "vote_count": m.get("vote_count"),
                    "original_language": m.get("original_language"),
                    "source": source,
                })
        print(f"Fetched {len(rows)} movies from the live TMDB API.")
    else:
        # FALLBACK path: no API key -> demonstrate the same pipeline on a
        # realistic sample payload in the exact TMDB JSON format.
        print("No TMDB API key found (set TMDB_API_KEY env var or create data/tmdb_api_key.txt).")
        print("Using bundled sample response data/tmdb_sample.json (same TMDB JSON format) ...")
        with open(TMDB_SAMPLE, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        for m in payload.get("results", []):
            rows.append({
                "title": m.get("title"),
                "release_date": m.get("release_date"),
                "rating": m.get("vote_average"),
                "popularity": m.get("popularity"),
                "vote_count": m.get("vote_count"),
                "original_language": m.get("original_language"),
                "source": "popular",
            })
        print(f"Loaded {len(rows)} movies from the sample payload.")

    # 4) convert the API response into a pandas DataFrame
    movies = pd.DataFrame(rows)

    # small clean-up of the collected columns
    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    movies["rating"] = pd.to_numeric(movies["rating"], errors="coerce")
    movies["popularity"] = pd.to_numeric(movies["popularity"], errors="coerce")
    movies = movies.dropna(subset=["title"]).reset_index(drop=True)

    print("\nTMDB movies DataFrame:")
    print(movies.head(10))
    print("\nShape:", movies.shape)

    # 5) save the collected data into tmdb_movies.csv
    movies.to_csv(TMDB_CSV, index=False)
    print(f"\nSaved collected data to data/tmdb_movies.csv ({len(movies)} rows)")
    return movies


# ==============================================================================
# PART 2 - DATA PREPROCESSING & CLEANING (Kaggle dataset from Task 1)
# ==============================================================================

# ------------------------------------------------------------------------------
# TASK 5: Understanding the data
# ------------------------------------------------------------------------------
def task5_understand_data(df):
    banner("PART 2 - TASK 5: UNDERSTANDING THE DATA")

    print("1. Dataset shape:", df.shape)
    print("\n2. Column data types:")
    print(df.dtypes)

    # 3. identify numerical vs categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print("\n3. Numerical columns   :", numerical_cols)
    print("   Categorical columns :", categorical_cols)

    # 4. missing values per column
    missing = df.isna().sum()
    print("\n4. Missing values per column:")
    print(missing[missing > 0].sort_values(ascending=False))
    print("   Total missing cells:", df.isna().sum().sum())

    print("\n5. Value counts of categorical columns:")
    for col in ["Sex", "Embarked", "Pclass", "Survived"]:
        print(f"\n   {col}:\n{df[col].value_counts()}")

    print("\n6. Unique values per column (cardinality):")
    print(df.nunique())
    return numerical_cols, categorical_cols


# ------------------------------------------------------------------------------
# TASK 6: Data cleaning
# ------------------------------------------------------------------------------
def task6_clean_data(df, products):
    banner("PART 2 - TASK 6: DATA CLEANING")
    df = df.copy()

    # --- 1. Handle missing values -------------------------------------------
    #    numerical  -> median (robust to outliers)
    #    categorical-> mode / "Unknown"
    print("Missing values BEFORE cleaning:")
    print(df.isna().sum().sum(), "total missing cells")
    print(df.isna().sum()[df.isna().sum() > 0])

    df["Age"] = df["Age"].fillna(df["Age"].median())          # numerical -> median
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # categorical -> mode
    df["Cabin"] = df["Cabin"].fillna("Unknown")               # categorical -> "Unknown"

    print("\nMissing values AFTER cleaning:", df.isna().sum().sum())

    # --- 2. Remove duplicate rows -------------------------------------------
    n_dupes = df.duplicated().sum()
    print("\nDuplicate rows found:", n_dupes)
    if n_dupes:
        df = df.drop_duplicates().reset_index(drop=True)
        print("Duplicates removed. New shape:", df.shape)

    # --- 3. Rename columns to lowercase snake_case ---------------------------
    rename_map = {
        "PassengerId": "passenger_id",
        "Survived": "survived",
        "Pclass": "pclass",
        "Name": "name",
        "Sex": "sex",
        "Age": "age",
        "SibSp": "sib_sp",
        "Parch": "parch",
        "Ticket": "ticket",
        "Fare": "fare",
        "Cabin": "cabin",
        "Embarked": "embarked",
    }
    df = df.rename(columns=rename_map)
    print("\nColumns renamed to lowercase snake_case:\n", list(df.columns))

    # --- 4. Fix incorrect data types -----------------------------------------
    print("\nData types BEFORE fixing:")
    print(df.dtypes)

    # survive / pclass are labels -> categorical codes are safer for modelling
    df["survived"] = df["survived"].astype("category")
    df["pclass"] = df["pclass"].astype("category")
    df["sex"] = df["sex"].astype("category")
    df["embarked"] = df["embarked"].astype("category")
    df["age"] = df["age"].astype("float64")
    df["fare"] = df["fare"].astype("float64")

    print("\nData types AFTER fixing (labels -> category, numbers -> float):")
    print(df.dtypes)

    # Bonus demo of string -> number fix using the JSON products dataset
    print("\n[Demo] Fixing string numbers in the products dataset (Task 2):")
    print("  BEFORE:", products[["price", "stock"]].dtypes.tolist())
    products["price"] = pd.to_numeric(products["price"], errors="coerce")
    products["stock"] = pd.to_numeric(products["stock"], errors="coerce")
    products["rating"] = pd.to_numeric(products["rating"], errors="coerce")
    print("  AFTER :", products[["price", "stock"]].dtypes.tolist())

    print("\nCleaned Titanic dataset - final shape:", df.shape)
    return df


# ------------------------------------------------------------------------------
# TASK 7: Feature preparation (basic)
# ------------------------------------------------------------------------------
def task7_feature_preparation(df):
    banner("PART 2 - TASK 7: FEATURE PREPARATION (BASIC)")
    df = df.copy()

    # 1a. Label Encoding for 'sex'  (binary column -> 0/1)
    df["sex_encoded"] = df["sex"].cat.codes            # male=1, female=0
    print("Label Encoding - 'sex' -> 'sex_encoded':")
    print(pd.crosstab(df["sex"], df["sex_encoded"]))

    # 1b. One-Hot Encoding for multi-class columns 'embarked' and 'pclass'
    embarked_dummies = pd.get_dummies(df["embarked"], prefix="embarked", drop_first=True)
    pclass_dummies = pd.get_dummies(df["pclass"], prefix="pclass", drop_first=True)
    print("\nOne-Hot Encoding - 'embarked':\n", embarked_dummies.head(3))
    print("\nOne-Hot Encoding - 'pclass':\n", pclass_dummies.head(3))

    df_encoded = pd.concat([df, embarked_dummies, pclass_dummies], axis=1)
    print("\nEncoded DataFrame shape:", df_encoded.shape)

    # 2. Separate features (X) and target (y)
    feature_cols = ["pclass", "sex_encoded", "age", "sib_sp", "parch", "fare",
                    "embarked_Q", "embarked_S"]
    target_col = "survived"
    X = df_encoded[feature_cols]
    y = df_encoded[target_col]

    print("\nFeatures X - shape:", X.shape)
    print(X.head())
    print("\nTarget y - shape:", y.shape, "| class distribution:")
    print(y.value_counts())
    return X, y


# ==============================================================================
# PART 3 - EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

# ------------------------------------------------------------------------------
# TASK 8: Univariate analysis
# ------------------------------------------------------------------------------
def task8_univariate(df):
    banner("PART 3 - TASK 8: UNIVARIATE ANALYSIS")

    numerical_cols = ["age", "fare", "sib_sp", "parch"]

    # 1. Histogram + KDE for numerical columns
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(df[col], kde=True, bins=30, ax=ax, color="#2e86de")
        ax.set_title(f"Distribution of {col} (histogram + KDE)")
        ax.set_xlabel(col)
        save_chart(fig, f"task8_hist_kde_{col}.png")

    # 2. Count plots for categorical columns
    for col in ["survived", "sex", "pclass", "embarked"]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.countplot(data=df, x=col, ax=ax, order=df[col].value_counts().index)
        ax.set_title(f"Count plot of {col}")
        for container in ax.containers:
            ax.bar_label(container)
        save_chart(fig, f"task8_countplot_{col}.png")

    # 3. Boxplots to identify outliers
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, col in zip(axes, numerical_cols):
        sns.boxplot(y=df[col], ax=ax, color="#f0932b")
        ax.set_title(f"Boxplot - {col}")
    save_chart(fig, "task8_boxplots_outliers.png")

    # IQR-based outlier report
    print("\nOutlier report (IQR method):")
    for col in numerical_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < low) | (df[col] > high)).sum()
        print(f"  {col:8s} -> outliers: {n_out:4d}  ({n_out / len(df):.1%} of rows)  "
              f"bounds=({low:.2f}, {high:.2f})")

    # 4. Pie charts (matplotlib) for proportions
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    df["sex"].value_counts().plot.pie(autopct="%1.1f%%", ax=axes[0],
                                      title="Passengers by sex", ylabel="")
    df["pclass"].value_counts().sort_index().plot.pie(autopct="%1.1f%%", ax=axes[1],
                                                      title="Passengers by class", ylabel="")
    save_chart(fig, "task8_piecharts.png")


# ------------------------------------------------------------------------------
# TASK 9: Bivariate analysis
# ------------------------------------------------------------------------------
def task9_bivariate(df):
    banner("PART 3 - TASK 9: BIVARIATE ANALYSIS")

    # --- 9a. Numerical vs Numerical ------------------------------------------
    # scatter plot: age vs fare, hue by survival
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.6, ax=ax)
    ax.set_title("Scatter plot: Age vs Fare (hue = Survived)")
    save_chart(fig, "task9_scatter_age_fare.png")

    # correlation heatmap of numerical columns
    num_cols = ["age", "fare", "sib_sp", "parch"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax)
    ax.set_title("Correlation heatmap - numerical features")
    save_chart(fig, "task9_correlation_heatmap.png")
    print("\nCorrelation matrix:\n", corr.round(2))

    # --- 9b. Categorical vs Numerical ----------------------------------------
    # bar plot: mean fare and mean age by passenger class
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    df.groupby("pclass", observed=True)["fare"].mean().plot.bar(
        ax=axes[0], color="#6ab04c", rot=0)
    axes[0].set_title("Mean fare by passenger class")
    axes[0].set_ylabel("mean fare")
    df.groupby("pclass", observed=True)["age"].mean().plot.bar(
        ax=axes[1], color="#eb4d4b", rot=0)
    axes[1].set_title("Mean age by passenger class")
    axes[1].set_ylabel("mean age")
    save_chart(fig, "task9_bar_mean_by_class.png")

    # box plot: age distribution by survival
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x="survived", y="age", ax=ax)
    ax.set_title("Box plot: Age by Survived")
    save_chart(fig, "task9_boxplot_age_by_survived.png")

    # violin plot: fare distribution by sex
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.violinplot(data=df, x="sex", y="fare", hue="survived", split=True, ax=ax)
    ax.set_title("Violin plot: Fare by Sex, split by Survived")
    save_chart(fig, "task9_violin_fare_by_sex.png")

    # stacked count plot: survival by sex and class
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.countplot(data=df, x="sex", hue="survived", ax=axes[0])
    axes[0].set_title("Survival by sex")
    sns.countplot(data=df, x="pclass", hue="survived", ax=axes[1])
    axes[1].set_title("Survival by passenger class")
    save_chart(fig, "task9_countplot_survival_by_sex_class.png")

    # survival-rate table used for the insights section
    print("\nSurvival rate by sex:\n",
          df.groupby("sex", observed=True)["survived"].apply(
              lambda s: f"{s.cat.codes.mean():.1%}").reset_index(name="survival_rate"))
    print("\nSurvival rate by pclass:\n",
          df.groupby("pclass", observed=True)["survived"].apply(
              lambda s: f"{s.cat.codes.mean():.1%}").reset_index(name="survival_rate"))


# ------------------------------------------------------------------------------
# TASK 10: Insights & observations
# ------------------------------------------------------------------------------
INSIGHTS_MD = """## PART 3 - TASK 10: Insights & Observations

Based on the EDA of the Kaggle **Titanic** dataset, here are the key insights:

1. **Gender is the strongest survival signal.** Females had a much higher survival
   rate than males, which is visible in both the survival-by-sex count plot and the
   violin plot (fare split by sex and survival).

2. **Socio-economic class strongly drives survival.** 1st-class passengers survived
   at a far higher rate than 3rd-class passengers - see the survival-by-pclass
   count plot and the mean-fare-by-class bar chart.

3. **Fare is right-skewed with extreme outliers.** The fare boxplot shows many
   points beyond the upper whisker (outliers detected with the IQR method). A few
   passengers paid several times the typical fare, which could distort a mean-based
   model - the median is the safer summary statistic here.

4. **Age contains a large block of missing values.** 177 of 891 ages were missing
   (~20%); they were imputed with the median in Task 6. Cabin was ~77% missing and
   was filled with "Unknown", making it of limited modelling value without extra
   feature work.

5. **Fare is moderately correlated with survival**, and it is tightly linked to
   passenger class (higher class -> higher fare). The correlation heatmap shows
   weak direct correlations among age/fare/sib_sp/parch, meaning the variables
   carry mostly independent information.

6. **Embarked and cabin are data-quality problem columns.** Embarked had 2 missing
   values (imputed with the mode 'S'); Cabin is mostly missing. Both need careful
   handling or feature engineering before any model is trained.

7. **The dataset is clean in most other respects.** No duplicate rows were found,
   and the numerical columns were stored in correct formats (only label columns
   were converted to `category` for clarity).

8. **Target class is imbalanced** (~62% did not survive, ~38% survived). Any future
   model should be evaluated with class imbalance in mind (e.g. stratified splits,
   precision/recall instead of accuracy alone).
"""


def task10_insights():
    banner("PART 3 - TASK 10: INSIGHTS & OBSERVATIONS (see INSIGHTS.md / README / notebook)")
    print(INSIGHTS_MD)

    path = os.path.join(BASE_DIR, "INSIGHTS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(INSIGHTS_MD)
    print(f"[saved] {path}")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    print("ASSIGNMENT 13 - DATA GATHERING, PREPROCESSING & EDA")
    print("Author: PATAN KHALILULLA KHAN")

    # PART 1 - Data gathering
    df_raw = task1_load_csv()
    products = task2_load_json()
    task3_load_sql()
    movies = task4_tmdb_api()

    # PART 2 - Preprocessing & cleaning
    task5_understand_data(df_raw)
    df_clean = task6_clean_data(df_raw, products)
    X, y = task7_feature_preparation(df_clean)

    # PART 3 - EDA
    task8_univariate(df_clean)
    task9_bivariate(df_clean)
    task10_insights()

    print("\n" + "#" * 100)
    print("PIPELINE COMPLETED SUCCESSFULLY.")
    print("Outputs:")
    print("  - data/sample.db        (SQLite database, Task 3)")
    print("  - data/tmdb_movies.csv  (TMDB movie data, Task 4)")
    print("  - INSIGHTS.md           (Task 10 insights)")
    print("  - charts/               (all EDA charts from Tasks 8 & 9)")
    print("#" * 100)


if __name__ == "__main__":
    main()
