# Customer Churn - Easy Guide to Fix Imbalanced Data

The notebook `imbalance_handle.ipynb` shows how to fix class imbalance in a churn dataset step by step. Every section uses simple language so you can follow the effect of each change on the K-Nearest Neighbors (KNN) model without needing advanced English.

## Files in this folder
- `customer_churn(in).csv` - source data (7,043 rows, 21 columns)
- `imbalance_handle.ipynb` - notebook with code, outputs, and notes
- `requirement.txt` - Python packages (`pandas`, `numpy`, `scikit-learn`, `imblearn`)

## Setup
1. Use Python 3.10 or newer and (optional) create a virtual environment.
2. Install the libraries: `pip install -r requirement.txt`
3. Place the CSV file in the same directory as the notebook.
4. Run `jupyter notebook imbalance_handle.ipynb` and execute all cells in order.

## Notebook workflow
1. **Explore and clean data**  
   Load the CSV, display `describe()` and `head()`, and verify there are no missing values. The 70% training split contains 3,641 "No" rows and 1,289 "Yes" rows, so the target is clearly imbalanced.
2. **Train/Test split (70/30)**  
   Split the dataframe before any balancing to avoid data leakage: `df_train` gets the first 70% of rows, `df_test` the remaining 30%.
3. **Feature builder**  
   Function `split_Xy` drops `Churn` from the feature set and keeps numeric columns only, which is required for KNN.
4. **Pipeline + Grid Search**  
   Build a pipeline with `StandardScaler` + `KNeighborsClassifier`, then run `GridSearchCV` (5-fold) across 9 neighbor counts and 2 weight schemes. This search is repeated for every training scenario so that each model uses tuned hyperparameters.
5. **Balancing strategies (train data only)**  
   - **Baseline** uses the raw `df_train` split.
   - **Fix size sampling** randomly samples the same number of "Yes" and "No" rows (1,289 each) before fitting.
   - **Oversampling by duplication** duplicates every minority example and appends it to the training set (resulting in 5,174 "No" and 4,447 "Yes").
   - **SMOTE** applies `SMOTE(random_state=42)` to `X_train`/`y_train` to create synthetic minority samples before running the same pipeline.
6. **Compare on the hold-out test set**  
   Each trained model predicts on `X_test`, and `classification_report` plus class counts show precision, recall, f1-score, and accuracy for the 30% hold-out data (1,533 "No", 580 "Yes").

## Evaluation summary (from the current notebook outputs)
All metrics below are computed on the 2,113-row test split with 1,533 "No" and 580 "Yes" customers.

| Method | Train adjustment | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) |
| --- | --- | --- | --- | --- | --- |
| Baseline | Original train split | 0.78 | 0.64 | 0.46 | 0.54 |
| Fix size sampling | Randomly sample 1,289 `Yes` and 1,289 `No` | 0.72 | 0.49 | 0.76 | 0.60 |
| Duplication oversampling | Duplicate every `Yes` row (5,174 `No`, 4,447 `Yes`) | 0.97 | 0.95 | 0.94 | 0.95 |
| SMOTE + KNN | `SMOTE(random_state=42)` on the train split | 0.71 | 0.48 | 0.61 | 0.54 |


## Next steps
1. Try other estimators (Logistic Regression, Random Forest, Gradient Boosting) inside the same pipeline to compare against KNN.
2. Tune the neighbor count, weight type, or SMOTE parameters (k-neighbors, sampling strategy) to trade off recall and precision for the churn class.
3. Export the `classification_report` tables or confusion matrices to CSV/plots to use in presentations or automated reports.
