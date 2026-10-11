# Daegu Residential Property Price Prediction

A machine learning project predicting residential property transaction prices in Daegu, South Korea, using 20 years of real government transaction records (2006-2024) across five housing types: Apartment, Single-family, Row house/Multiplex, Multi-family, and Officetel.

## Business Question

What does a residential property in Daegu sell for, given its location, size, age, and housing type, and which of these factors matter most in predicting that price?

## Model Performance

A tuned **XGBoost** regressor, selected over Random Forest (Wilcoxon signed-rank test across 10 time-series folds, p = 0.0059) and a linear baseline (R² = 0.707, ruled out due to severe multicollinearity and non-linear relationships it could not capture). Evaluated on a held-out 2023-2024 test set (52,160 transactions never seen during training or tuning):

| Metric | Overall |
|---|---|
| RMSE | 12,084 (10,000 KRW) |
| MAE | 5,881 (10,000 KRW) |
| MAPE | 19.6% |

## Key Findings

- **Property size and transaction year are the strongest, most consistent price drivers**, confirmed across multiple feature importance metrics, not just one.
- **Accuracy varies sharply by housing type**: Officetel is predicted most reliably (13.8% MAPE), likely due to its standardized unit sizes, while Single-family is the weakest (62.8% MAPE), driven by two distinct causes: a scarcity of large, high-value training examples, and small central-district properties whose redevelopment value isn't captured by district-level location alone.
- **Small, older properties in central districts carry a price premium per square meter**, the opposite of what a simple linear size-to-price relationship would predict, this non-linearity is a large part of why a tree-based model outperforms a linear one.
- **Feature importance rankings depend on the metric used**: categorical features (housing type) score highly on `gain` (large, decisive splits), while continuous features (location encoding) score highly on `weight` and `cover` (frequent, incremental use), both are real contributions, measured differently.

## Recommendations

1. The model is most reliable for Apartment transactions (74% of the training data), and can reasonably support automated or semi-automated price estimation for this segment.
2. For Single-family properties, especially small properties in central districts or unusually large ones, treat predictions as a rough reference only, not a substitute for manual appraisal.
3. Future iterations would benefit most from precise property coordinates, to replace the current district-level location granularity, and a disambiguated building-vs-unit area field for Single-family/Multi-family transactions.
4. Given how differently housing types behave, a future version could test housing-type-specific models rather than one shared model.

## Visuals

![XGBoost Feature Importance](reports/figures/feature_importance.png)
![Price by Housing Type](reports/figures/price_by_housing_type.png)
![Actual vs. Predicted Price](reports/figures/actual_vs_predicted.png)

## Data

This project uses Daegu residential property transaction records (August 2005 to December 2024, 740,480 transactions), sourced from Korea's Ministry of Land, Infrastructure and Transport, Actual Transaction Price Disclosure System, via the Public Data Portal. Address-level identifiers (lot numbers, road name, building name) were excluded, as they don't generalize to properties the model hasn't seen. Columns with structural missingness tied to specific housing types or a 2021 disclosure policy change (`land_area_m2`, `land_ownership_area_m2`, `transaction_type`) were also excluded rather than imputed. Full column definitions are in [`data/data_dictionary.md`](data/data_dictionary.md).

**License:** [KOGL Type 1](https://www.kogl.or.kr/info/license.do), free to use, redistribute, and adapt, with attribution required. Independently compiled, not affiliated with or endorsed by the Ministry of Land, Infrastructure and Transport.

**Getting the full data:** the raw file (99.5 MB) is not included in this repository, GitHub's browser upload rejects files over 25 MB. Download it from [Kaggle: daegu apartment actual transaction](https://www.kaggle.com/datasets/lnoahl/daegu-aptmernt-actual-transaction) and place it at `data/raw/Daegu_apt.csv` before running the notebook.

## Limitations

This analysis covers 20 years of transactions but location is known only at the district and sub-locality level, no coordinates are available, which limits the model's ability to price individual parcels precisely and is the primary driver of the Single-family segment's weaker performance. Prices are used as recorded and are not adjusted for inflation or macroeconomic conditions across the period, so the model reflects nominal price patterns, not real, policy-adjusted ones. Whether unusually large Single-family/Multi-family transactions represent legitimate whole-building sales or a recording inconsistency could not be resolved with the columns available.

## Tools

Python, pandas, numpy, scikit-learn, xgboost, statsmodels, scipy, matplotlib, seaborn, Jupyter Notebook

## How to Run

1. Clone or download this repository
2. Obtain the raw dataset and place it at `data/raw/Daegu_apt.csv` (see **Data** above)
3. Create and activate a virtual environment:
   `python -m venv venv` then `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Open `notebooks/Daegu_Apartment_Price_Prediction.ipynb` in Jupyter or VS Code, and select the `venv` kernel
6. Run all cells

## Repository Structure

```
Daegu-Apartment-Price-Prediction/
├── data/
│ ├── raw/
│ |   └── Daegu_apt.csv      (not included, see Data section)
│ ├── processed/             (generated by running the notebook)
│ └── data_dictionary.md
├── models/
│ └── final_model.pkl
├── notebooks/
│ └── Daegu_Apartment_Price_Prediction.ipynb
├── reports/
│ └── figures/
├── README.md
└── requirements.txt
```

## Full Analysis

See the full notebook for detailed methodology, data cleaning steps, and findings for each section: notebooks/Daegu_Apartment_Price_Prediction.ipynb
