# Daegu Apartment Fair Price Prediction (Machine Learning Regression)

A data-driven machine learning project designed to predict fair and competitive apartment sale prices in Daegu, South Korea. This project utilizes rigorous data preprocessing, feature engineering, and advanced regression modeling to deliver objective valuation tools for property owners and real estate agents.

---

## Project Overview
In a bustling metropolitan area like Daegu, setting the right price for residential properties is critical. Overpricing causes units to stagnate on the market, while underpricing results in substantial financial loss for sellers. 

This project solves this dilemma by building a predictive regression model that analyzes a combination of internal attributes (e.g., unit size, build year) and external location factors (e.g., proximity to public transit, neighborhood facilities) to determine a property's fair market value.

---

## Key Features & Analytic Approach

The end-to-end Machine Learning pipeline implemented in this project follows a structured methodology:

### 1. Business & Data Understanding
* Evaluated key valuation drivers based on historical apartment transaction data in Daegu.
* Defined targeted business metrics to align model evaluation with real-world financial risk.

### 2. Preprocessing & Feature Engineering
* **Data Cleaning**: Managed missing entries, handled anomalous values, and removed structural noise.
* **Feature Transformation**: Applied categorical encoding (`OneHotEncoder` via pipelines) and scaled numerical boundaries where necessary to prevent bias.
* **Feature Selection**: Isolated critical variables such as `HallwayType`, `TimeToSubway`, `SubwayStation`, and physical asset metrics.

### 3. Machine Learning Modeling
* **Baseline Models**: Established initial benchmarks using standard models like **Linear Regression** and **Random Forest Regressor**.
* **Final Champion Model**: Implemented **XGBoost Regressor**, optimized via comprehensive hyperparameter tuning to minimize prediction errors and avoid overfitting.

### 4. Robust Evaluation Metrics
* Evaluated using multiple standard metrics to capture different aspects of prediction errors: **MAE** (Mean Absolute Error), **RMSE** (Root Mean Squared Error), **MAPE** (Mean Absolute Percentage Error), and **R-squared ($R^2$)**.

---

## Dataset & Attributes

The target variable for this regression task is **`SalePrice`** (Apartment Market Value). The model leverages diverse architectural and geographical features:

| Attribute | Category | Description |
| :--- | :--- | :--- |
| **Physical Attributes** | Internal | Unit size (sqft), Year built, Number of basement parking slots. |
| **Infrastructure** | External | Hallway type (`HallwayType`), Closest subway station name (`SubwayStation`). |
| **Accessibility** | External | Precise travel time or distance to the nearest subway line (`TimeToSubway`). |
| **Environment** | External | Total volume of public offices, universities, and general facilities nearby. |

---

## Key Outcomes & Model Performance
* The finalized **XGBoost Regressor** model demonstrated superior generalizing capabilities during validation, significantly reducing both absolute and percentage errors compared to standard baselines.
* The model pipelines and structural weights were successfully compiled and exported into a binary deployment package (**`xgboost_daegu_apartment_model.pkl`**) via Python’s `pickle` library, ensuring seamless serialization for future application integration.

---

## Strategic Business Recommendations

### 1. Deployment for Real Estate Portals
* Integrate the serialized `.pkl` model into an interactive pricing tool for real estate agents or secondary market sellers to automatically recommend competitive initial listing prices based on objective asset data.

### 2. Strategic Boundaries & Data Scope Limitations
* **Handling Unique Units**: The model performs optimally on standardized apartment formats. Its predictive accuracy may fluctuate on ultra-rare luxury penthouses or legacy units that exhibit unique historical outliers not captured in the current training set.

### 3. Future Enhancements & Feature Expansion
* **Dynamic Valuation**: Incorporate live, macro-economic feeds (e.g., real-time interest rate fluctuations, local inflation markers).
* **Geospatial Enrichment**: Integrate richer spatial dimensions such as neighborhood zoning maps, school district tier rankings, or land-value indexing (`Nilai Jual Objek Pajak` equivalent) to further stabilize price predictions.

---

## Requirements & Technical Stack

The core ecosystem for this project is entirely built in Python:

* **Data Wrangling**: `pandas`, `numpy`
* **Visualization**: `matplotlib`, `seaborn`
* **Machine Learning & Pipeline**: `scikit-learn`
* **Gradient Boosting Framework**: `xgboost`
* **Model Serialization**: `pickle`
* **Environment**: Jupyter Notebook (`.ipynb`)
