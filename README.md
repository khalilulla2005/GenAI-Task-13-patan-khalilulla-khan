# GenAI-Task(13) — Data Gathering, Preprocessing & EDA

> **Author:** PATAN KHALILULLA KHAN
> **Course assignment 13:** Data Gathering, Preprocessing & Exploratory Data Analysis
> **Scope:** Pre-ML foundations only — **no model training**, per the problem statement.
> **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `requests`, `sqlite3` (stdlib) — nothing else.

---

## Overview

This repository implements the complete pre-ML pipeline across **10 tasks in 3 parts**:

| Part | Tasks | Focus |
|------|-------|-------|
| **1 — Data Gathering** | 1–4 | CSV (Kaggle), JSON, SQLite database, TMDB REST API |
| **2 — Preprocessing & Cleaning** | 5–7 | Data understanding, cleaning, encoding & feature/target split |
| **3 — EDA** | 8–10 | Univariate analysis, bivariate analysis, written insights |

Every task prints **real output** — the exact results shown below are produced by
running `python main.py` on the checked-in datasets, and every chart is saved as a
PNG in `charts/`.

---

## Quick start

```bash
pip install -r requirements.txt
python main.py            # run all 10 tasks end-to-end
python test_assignment13.py   # run the 52-check automated test suite
```

---

## PART 1 — Data Gathering (real outputs)

### Task 1 — Load data from CSV (Kaggle Titanic)

Dataset: **Titanic — Machine Learning from Disaster** (Kaggle), `data/titanic.csv`.

```text
Shape of the dataset: (891, 12)
Column names: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

First 5 rows:
 PassengerId  Survived  Pclass                                                Name    Sex  Age  SibSp  Parch           Ticket    Fare Cabin Embarked
           1         0       3                             Braund, Mr. Owen Harris   male 22.0      1      0        A/5 21171  7.2500   NaN        S
           2         1       1 Cumings, Mrs. John Bradley (Florence Briggs Thayer) female 38.0      1      0         PC 17599 71.2833   C85        C
           3         1       3                              Heikkinen, Miss. Laina female 26.0      0      0 STON/O2. 3101282  7.9250   NaN        S
           4         1       1        Futrelle, Mrs. Jacques Heath (Lily May Peel) female 35.0      1      0           113803 53.1000  C123        S
           5         0       3                            Allen, Mr. William Henry   male 35.0      0      0           373450  8.0500   NaN        S

Numerical summary:
          Age    Fare
count  714.00  891.00
mean    29.70   32.20
std     14.53   49.69
min      0.42    0.00
25%     20.12    7.91
50%     28.00   14.45
75%     38.00   31.00
max     80.00  512.33
```

### Task 2 — Load data from JSON

Hand-made product catalogue `data/products.json` → DataFrame via `pd.read_json`.
`price`/`stock` are deliberately kept as **strings** (dtype `object`) to be fixed in
Task 6.

```text
products.json -> DataFrame (shape (8, 7))
 product_id                  name    category price stock  rating  in_stock
          1        Wireless Mouse Electronics 19.99   120     4.5      True
          2   Mechanical Keyboard Electronics 59.50    45     4.8      True
          3           USB-C Cable Accessories  7.25   300     4.1      True
          4          Laptop Stand Accessories 29.99     0     4.3     False
          5             HD Webcam Electronics 49.00    18     4.6      True
          6             Desk Lamp        Home 15.75    60     4.0      True
          7          Notebook Set  Stationery  9.99   150     NaN      True
          8 Gel Pens (Pack of 10)  Stationery  5.49   200     4.2      True

dtypes:
product_id      int64
name           object
category       object
price          object
stock          object
rating        float64
in_stock         bool
```

### Task 3 — Load data from SQL database (SQLite)

`data/sample.db` is created **in code**: the `employees` table is built, 8 records
are inserted, and the data is read back into pandas with a SQL query.

```text
employees table (read via SQL query, shape (8, 4)):
 id                  name      department  salary
  8          Meena Kumari     Engineering 75000.0
  2       Ayesha Siddiqua     Engineering 72000.0
  3           Rahul Verma     Engineering 68000.0
  7           Arjun Reddy    Data Science 64000.0
  5          Vikram Singh         Finance 61000.0
  1 Patan Khalilulla Khan    Data Science 55000.0
  6            Sana Patel       Marketing 52000.0
  4           Fatima Noor Human Resources 48000.0

SQL aggregation - avg salary per department:
     department  n_employees  avg_salary
    Engineering            3    71666.67
        Finance            1    61000.00
   Data Science            2    59500.00
      Marketing            1    52000.00
Human Resources            1    48000.00
```

### Task 4 — API mini-project: TMDB movie data collector

`requests` → TMDB API → DataFrame → `data/tmdb_movies.csv` (title, release date,
rating, popularity, vote count, language).

```text
tmdb_movies.csv (shape (10, 7)) - first 5 movies:
                          title release_date  rating  popularity  vote_count original_language  source
                 Dune: Part Two   2024-02-27   8.171    2973.436        5060                en popular
                Kung Fu Panda 4   2024-03-02   7.108    1976.032        2359                en popular
                      No Way Up   2024-01-18   6.262    1314.948         728                en popular
                 Code 8 Part II   2024-02-27   6.886    1120.128         623                en popular
Godzilla x Kong: The New Empire   2024-03-27   7.274    1082.438         892                en popular
```

> **Live API mode:** set `TMDB_API_KEY` (env var) or drop the key into
> `data/tmdb_api_key.txt` — the script then fetches **popular + top-rated** movies
> from the real endpoint. Without a key it uses `data/tmdb_sample.json`, a payload
> in the exact TMDB JSON format, so the pipeline always runs.

---

## PART 2 — Data Preprocessing & Cleaning (real outputs)

### Task 5 — Understanding the data

```text
Shape: (891, 12)
Numerical columns : ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
Categorical columns: ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

Missing values per column:
Cabin       687
Age         177
Embarked      2

Value counts - Survived:
Survived
0    549
1    342

Value counts - Embarked:
Embarked
S    644
C    168
Q     77
```

### Task 6 — Data cleaning

Missing values: `Age` → **median**, `Embarked` → **mode**, `Cabin` → **"Unknown"**;
duplicates removed (0 found); columns renamed to **lowercase snake_case**; dtypes
fixed (labels → `category`, numbers → `float64`, string numbers → numeric).

```text
Missing values BEFORE: 866 | AFTER: 0
Duplicates: 0
Renamed columns: ['passenger_id', 'survived', 'pclass', 'name', 'sex', 'age', 'sib_sp', 'parch', 'ticket', 'fare', 'cabin', 'embarked']

dtypes after fixing:
passenger_id       int64
survived        category
pclass          category
name              object
sex             category
age              float64
sib_sp             int64
parch              int64
ticket            object
fare             float64
cabin             object
embarked        category
```

### Task 7 — Feature preparation

Label encoding for `sex`, one-hot encoding for `embarked`/`pclass`, then the
feature/target split (`survived` is the target).

```text
Features X shape: (891, 8) | Target y: (891,)

X.head():
  pclass  sex_encoded   age  sib_sp  parch     fare  embarked_Q  embarked_S
0      3            1  22.0       1      0   7.2500       False        True
1      1            0  38.0       1      0  71.2833       False       False
2      3            0  26.0       0      0   7.9250       False        True
3      1            0  35.0       1      0  53.1000       False        True
4      3            1  35.0       0      0   8.0500       False        True

Target distribution:
survived
0    549
1    342
```

---

## PART 3 — Exploratory Data Analysis

### Task 8 — Univariate analysis (16 charts in `charts/`)

Histograms + KDE for all numerical columns, count plots for all categorical
columns, boxplots + IQR outlier report, pie charts.

```text
Outlier report (IQR method):
  age      -> outliers:   66  (7.4% of rows)  bounds=(2.50, 54.50)
  fare     -> outliers:  116  (13.0% of rows)  bounds=(-26.72, 65.63)
  sib_sp   -> outliers:   46  (5.2% of rows)  bounds=(-1.50, 2.50)
  parch    -> outliers:  213  (23.9% of rows)  bounds=(0.00, 0.00)
```

### Task 9 — Bivariate analysis

Scatter plots, correlation heatmap, bar/box/violin plots, stacked count plots.

```text
Correlation matrix:
          age  fare  sib_sp  parch
age     1.00  0.10   -0.23  -0.17
fare    0.10  1.00    0.16   0.22
sib_sp -0.23  0.16    1.00   0.41
parch  -0.17  0.22    0.41   1.00

Survival rate by sex:
       sex survival_rate
0  female         74.2%
1    male         18.9%

Survival rate by pclass:
   pclass survival_rate
0      1         63.0%
1      2         47.3%
2      3         24.2%
```

### Task 10 — Insights & observations

Full text in [`INSIGHTS.md`](INSIGHTS.md); the eight key insights:

1. **Gender is the strongest survival signal** — 74.2% of females survived vs 18.9% of males.
2. **Socio-economic class strongly drives survival** — 63.0% (1st) vs 24.2% (3rd class).
3. **Fare is right-skewed with extreme outliers** — 116 fare outliers (IQR method); use the median.
4. **Age has ~20% missing values** (median-imputed); **Cabin is ~77% missing** (→ "Unknown").
5. **Fare is moderately correlated with survival** and tightly linked to class.
6. **Embarked and Cabin are the main data-quality problem columns.**
7. **No duplicate rows**; remaining data was already in correct formats.
8. **Target is imbalanced** (~62% / 38%) — future models should use stratified splits and precision/recall.

---

## Automated test suite

`test_assignment13.py` runs **52 checks** covering all 10 tasks (file existence,
shapes, column names, dtypes, missing-value handling, encodings, charts, insights).
The last verified run (log: [`test_results.txt`](test_results.txt)):

```text
TOTAL: 52 checks  |  PASSED: 52  |  FAILED: 0
ALL TEST CASES PASSED
(exit code: 0)
```

---

## Repository structure

```
├── main.py                  # full pipeline, Tasks 1-10
├── Assignment13_EDA.ipynb   # executed notebook (insights in markdown cells)
├── test_assignment13.py     # 52-check automated test suite
├── test_results.txt         # log of the last test run (52/52 PASSED)
├── INSIGHTS.md              # Task 10 insights (markdown)
├── README.md
├── requirements.txt
├── data/
│   ├── titanic.csv          # Kaggle dataset (Task 1)
│   ├── products.json        # hand-made JSON (Task 2)
│   ├── sample.db            # SQLite database (Task 3)
│   ├── tmdb_movies.csv      # collected movie data (Task 4)
│   ├── tmdb_sample.json     # TMDB-format fallback payload
│   └── tmdb_api_key.txt     # (optional) your TMDB API key for live mode
└── charts/                  # 16 EDA charts, PNG (Tasks 8 & 9)
```
