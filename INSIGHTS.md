## PART 3 - TASK 10: Insights & Observations

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
