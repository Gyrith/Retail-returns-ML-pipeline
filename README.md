# Retail Returns ML Pipeline
Machine learning pipeline predicting which retail products are likely to have high return rates, using preprocessing pipelines and comparing Random Forest vs. Logistic Regression.

## Contents
- `retail_return_model_lab.ipynb` — full notebook: business framing, EDA, preprocessing pipeline, model comparison via GridSearchCV, final evaluation, and feature importance.
- `product_return_data.csv` — dataset used by the notebook.

## Business Context
Product returns are costly (processing costs of $15-25/item), but overly restricting the catalog to avoid returns costs lost sales. The model is optimized for **F1 score** to balance precision and recall on identifying high-return products.

## Approach
1. Define business goals and primary metric (F1)
2. Explore the data (distributions, missingness, class balance)
3. Build preprocessing (imputation, scaling, one-hot encoding) via `ColumnTransformer`
4. Build full pipelines combining preprocessing + model
5. Tune Random Forest and Logistic Regression via `GridSearchCV`
6. Evaluate the best model (test F1, classification report, confusion matrix) and interpret feature importances

## Results
- Best model: **Random Forest** (CV F1 = 0.914, test F1 = 0.916)

## Tech
Python, scikit-learn, pandas, matplotlib, seaborn