"""
===============================================================================
 Assignment 13 - TEST SUITE (all 10 tasks)
 Author  : PATAN KHALILULLA KHAN
 Purpose : Automated verification that every task produces the expected output.
           Run with:  python test_assignment13.py
===============================================================================
"""

import os
import sqlite3

import pandas as pd
import matplotlib
matplotlib.use("Agg")

import main  # the pipeline module

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


print("=" * 90)
print("TEST SUITE - ASSIGNMENT 13 (PATAN KHALILULLA KHAN)")
print("=" * 90)

# ----------------------------------------------------------------------------
# PART 1 - Data gathering
# ----------------------------------------------------------------------------
print("\n[PART 1 - DATA GATHERING]")

print("\n  TASK 1: Load data from CSV")
df_raw = main.task1_load_csv()
check("CSV file exists", os.path.exists(main.CSV_PATH))
check("Shape is (891, 12)", df_raw.shape == (891, 12), f"got {df_raw.shape}")
expected_cols = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
                 "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
check("Column names match Titanic schema", list(df_raw.columns) == expected_cols)
check("No empty rows", df_raw.empty is False)

print("\n  TASK 2: Load data from JSON")
products = main.task2_load_json()
check("JSON file exists", os.path.exists(main.JSON_PATH))
check("products has 8 rows", len(products) == 8, f"got {len(products)}")
check("price column is string dtype (before cleaning)",
      products["price"].dtype == object)
check("has required columns",
      set(["product_id", "name", "category", "price", "stock", "rating"])
      <= set(products.columns))

print("\n  TASK 3: Load data from SQL database")
main.task3_load_sql()
check("sample.db created", os.path.exists(main.DB_PATH))
conn = sqlite3.connect(main.DB_PATH)
emp = pd.read_sql_query("SELECT * FROM employees", conn)
conn.close()
check("employees table has >= 5 records", len(emp) >= 5, f"got {len(emp)}")
check("required columns present",
      set(["id", "name", "department", "salary"]) <= set(emp.columns))

print("\n  TASK 4: TMDB API mini project")
main.task4_tmdb_api()
check("tmdb_movies.csv saved", os.path.exists(main.TMDB_CSV))
movies = pd.read_csv(main.TMDB_CSV)
check("movies DataFrame is not empty", len(movies) > 0)
check("extracted title column", "title" in movies.columns)
check("extracted release_date column", "release_date" in movies.columns)
check("extracted rating column", "rating" in movies.columns)
check("extracted popularity column", "popularity" in movies.columns)
check("ratings are numeric", pd.api.types.is_numeric_dtype(movies["rating"]))

# ----------------------------------------------------------------------------
# PART 2 - Preprocessing & cleaning
# ----------------------------------------------------------------------------
print("\n[PART 2 - DATA PREPROCESSING & CLEANING]")

print("\n  TASK 5: Understanding the data")
num_cols, cat_cols = main.task5_understand_data(df_raw)
check("Age has 177 missing values",
      df_raw["Age"].isna().sum() == 177, f"got {df_raw['Age'].isna().sum()}")
check("Cabin has 687 missing values",
      df_raw["Cabin"].isna().sum() == 687, f"got {df_raw['Cabin'].isna().sum()}")
check("Embarked has 2 missing values",
      df_raw["Embarked"].isna().sum() == 2, f"got {df_raw['Embarked'].isna().sum()}")
check("numerical/categorical split correct",
      "Sex" in cat_cols and "Age" in num_cols)

print("\n  TASK 6: Data cleaning")
df_clean = main.task6_clean_data(df_raw, products)
check("No missing values remain", df_clean.isna().sum().sum() == 0)
check("No duplicate rows", df_clean.duplicated().sum() == 0)
check("Columns renamed to snake_case",
      list(df_clean.columns) == ["passenger_id", "survived", "pclass", "name",
                                 "sex", "age", "sib_sp", "parch", "ticket",
                                 "fare", "cabin", "embarked"])
check("survived is categorical dtype",
      pd.api.types.is_categorical_dtype(df_clean["survived"]))
check("age is float64", df_clean["age"].dtype == "float64")
# string -> number demo on products
prods = products.copy()
prods["price"] = pd.to_numeric(prods["price"], errors="coerce")
check("products price converted string -> float",
      pd.api.types.is_numeric_dtype(prods["price"]))

print("\n  TASK 7: Feature preparation")
X, y = main.task7_feature_preparation(df_clean)
check("X has 8 feature columns", X.shape[1] == 8, f"got {X.shape[1]}")
check("y is the survived target", y.name == "survived")
check("sex label-encoded present", "sex_encoded" in X.columns)
check("embarked one-hot columns present",
      set(["embarked_Q", "embarked_S"]) <= set(X.columns))
check("X/y have same row count", len(X) == len(y))

# ----------------------------------------------------------------------------
# PART 3 - EDA
# ----------------------------------------------------------------------------
print("\n[PART 3 - EXPLORATORY DATA ANALYSIS]")

print("\n  TASK 8: Univariate analysis")
main.task8_univariate(df_clean)
charts8 = ["task8_hist_kde_age.png", "task8_hist_kde_fare.png",
           "task8_hist_kde_sib_sp.png", "task8_hist_kde_parch.png",
           "task8_countplot_survived.png", "task8_countplot_sex.png",
           "task8_countplot_pclass.png", "task8_countplot_embarked.png",
           "task8_boxplots_outliers.png", "task8_piecharts.png"]
for c in charts8:
    check(f"chart saved: {c}", os.path.exists(os.path.join(main.CHART_DIR, c)))

print("\n  TASK 9: Bivariate analysis")
main.task9_bivariate(df_clean)
charts9 = ["task9_scatter_age_fare.png", "task9_correlation_heatmap.png",
           "task9_bar_mean_by_class.png", "task9_boxplot_age_by_survived.png",
           "task9_violin_fare_by_sex.png",
           "task9_countplot_survival_by_sex_class.png"]
for c in charts9:
    check(f"chart saved: {c}", os.path.exists(os.path.join(main.CHART_DIR, c)))

print("\n  TASK 10: Insights & observations")
main.task10_insights()
check("INSIGHTS.md written", os.path.exists(os.path.join(main.BASE_DIR, "INSIGHTS.md")))
insights_text = main.INSIGHTS_MD
check("at least 5 insights written",
      sum(1 for line in insights_text.splitlines()
          if line.strip()[:2] in [f"{i}." for i in range(1, 21)]) >= 5)
check("README.md exists", os.path.exists(os.path.join(main.BASE_DIR, "README.md")))

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
print("\n" + "=" * 90)
print(f"TOTAL: {PASS + FAIL} checks  |  PASSED: {PASS}  |  FAILED: {FAIL}")
if FAIL == 0:
    print("ALL TEST CASES PASSED")
else:
    print(f"{FAIL} TEST CASE(S) FAILED")
print("=" * 90)
raise SystemExit(0 if FAIL == 0 else 1)
