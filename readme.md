# Customer Churn – Easy Guide to Fix Imbalanced Data

This Jupyter Notebook (`imbalance_handle.ipynb`) teaches how to spot and fix class imbalance in a churn dataset. The focus is on keeping the steps simple so you can see how each change helps the model catch customers who will churn.

## Files in this folder
- `customer_churn(in).csv` – source data (7,043 rows, 21 columns).
- `imbalance_handle.ipynb` – notebook with every step.
- `requirement.txt` – libraries to install (`pandas`, `numpy`, `scikit-learn`, `imblearn`).

## Quick start
1. Use Python 3.10+ and (optional) create a virtual environment.
2. Install packages: `pip install -r requirement.txt`
3. Put the CSV in the same folder.
4. Run `jupyter notebook imbalance_handle.ipynb` and execute the cells from top to bottom.

## What the notebook does
1. **Explore and clean data**  
   - Load the CSV, check summary stats with `describe()`, preview rows with `head()`, and count missing values.  
   - After `dropna()` the training set has 3,641 “No” rows and 1,289 “Yes” rows, so the target is heavily imbalanced.

2. **Train/Test split (70/30)**  
   - `df_train` contains the first 70% of rows and `df_test` keeps the rest.  
   - Splitting before any balancing avoids data leakage.

3. **Feature builder**  
   - Function `split_Xy` removes the `Churn` column and keeps only numeric columns. KNN needs numbers only.

4. **Pipeline + Grid Search**  
   - Pipeline = `StandardScaler` + `KNeighborsClassifier`.  
   - `GridSearchCV` tries 9 values of `n_neighbors` × 2 weight settings (uniform/distance) with 5-fold CV to find the best combo for every scenario.

5. **Three balancing strategies (train set only)**  
   - **Baseline**: use raw `df_train`. Shows how imbalance hurts the minority class.  
   - **Fix size sampling**: randomly sample the same count of `Yes` and `No` (1,289 each) to train a balanced model.  
   - **Oversampling by duplication**: duplicate all `Yes` rows and add them back to `df_train`, so the minority class is larger.  
   - **SMOTE**: apply `SMOTE(random_state=42)` to `X_train`/`y_train`, then fit the same pipeline. This cell requires `imblearn`; install from `requirement.txt` first.

6. **Compare models on the hold-out test set**  
   - Every model predicts on `X_test`, and the notebook prints `classification_report` so you can compare precision, recall, f1, and accuracy.

## Sample results (re-run locally)
All numbers below evaluate on the 30% test split (1,533 “No”, 580 “Yes”).

| Method | Best params | Accuracy | f1 (Churn = Yes) | Notes |
| --- | --- | --- | --- | --- |
| Baseline | `n_neighbors=11`, `weights='uniform'` | 0.78 | 0.54 | Good for “No”, weak recall for “Yes”. |
| Fix size sampling | same as baseline | 0.72 | 0.60 | Recall for “Yes” jumps to 0.76 at the cost of “No”. |
| Duplication oversampling | `n_neighbors=1`, `weights='uniform'` | 0.72 | 0.48 | Keeps accuracy but still struggles to lift recall. |
| SMOTE + KNN | (run notebook cell after installing `imblearn`) | — | — | Creates synthetic “Yes” rows; check the report in the notebook for the exact numbers. |

## Tips for further work
1. Try other models (Logistic Regression, Random Forest) with the same pipeline idea.
2. Tune `n_neighbors`, weight type, or SMOTE settings to search for better recall vs. precision balance.
3. Export the `classification_report` outputs to CSV or plots for presentations.

