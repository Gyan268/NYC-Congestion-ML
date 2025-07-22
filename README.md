# NYC Congestion Relief Zone Analysis with Machine Learning

## Overview
This project explores NYC's **Congestion Relief Zone (CRZ)** data to analyze traffic congestion patterns and propose data-driven tolling strategies.  
Using a dataset of **~650,000 records**, I apply **binary classification models** to predict whether congestion will be **High (1)** or **Low (0)** based on historical traffic patterns.

Beyond modeling, I recommend an **optimized pricing schedule** for congestion reduction, based on insights from CART and Random Forest models.

---

## Goals
1. **Reduce Congestion:** Encourage drivers to adjust travel behavior via dynamic pricing.
2. **Generate Funding:** CRZ tolling generated $51.9M in February 2024, supporting MTA infrastructure upgrades.

---

## Dataset
**Source:** NYC CRZ entry logs  
**Size:** ~650,000 rows after cleaning

**Key Features:**
- **Toll Week:** Timestamp for the week of data collection.
- **Hour of Day (0-23) & Minute of Hour:** Temporal features.
- **Day of Week:** Categorical (Monday–Sunday).
- **Detection Region:** Entry sensor location.
- **Vehicle Class:** Car, motorcycle, truck, etc.
- **Target Variable:** Binary label — **1 = High Congestion** if CRZ entries ≥ median, **0 = Low Congestion**.

**Preprocessing:**
- One-hot encoding for categorical features (Vehicle Class, Day of Week).
- Removal of redundant columns (e.g., Toll Hour, Time Period).
- Dropped ~2% missing values.

---

## Machine Learning Models
I implemented and evaluated **8 models**, each in its own Jupyter Notebook (see `notebooks/`):

1. **Naive Bayes** – Baseline model using Gaussian assumptions.
2. **Artificial Neural Network (ANN)** – Multi-layer perceptron capturing nonlinear patterns.
3. **CART (Decision Trees)** – Interpretable baseline model.
4. **K-Nearest Neighbors (KNN)** – Distance-based classification.
5. **Support Vector Machine (SVM)** – Margin-based classification for high precision.
6. **Random Forest** – Ensemble of decision trees, robust to noise.
7. **Clustering (KMeans)** – Unsupervised grouping of traffic patterns.
8. **Isolation Forest** – Outlier detection for unusual traffic spikes.

---

## Results Summary
- **Naive Bayes:** A simple, fast baseline but less accurate for peak-hour detection.
- **ANN:** Best performance, accurately captures nonlinear peak patterns.
- **Random Forest & SVM:** Both achieved high precision/recall, ideal for dynamic tolling.
- **CART:** Balanced accuracy and interpretability, forming the basis for our pricing recommendation.
- **KNN:** High accuracy but slower prediction.
- **Isolation Forest:** Found no significant anomalies, confirming data stability.
- **Clustering:** Not suitable for this binary classification task.

### Model Performance Comparison

| Model                           | Accuracy (%)   | Precision (%)   | Recall (%)   |
|:--------------------------------|:---------------|:----------------|:-------------|
| Naive Bayes                     | 78             | 77              | 78           |
| Artificial Neural Network (ANN) | 92             | 93              | 91           |
| CART                            | 89             | 88              | 90           |
| K-Nearest Neighbors (KNN)       | 88             | 87              | 88           |
| Support Vector Machine (SVM)    | 90             | 91              | 89           |
| Random Forest                   | 91             | 90              | 92           |
| Clustering (KMeans)             | 65             | N/A             | N/A          |
| Isolation Forest                | N/A            | N/A             | N/A          |

### Visuals (Examples)
- Confusion matrices and bar charts for actual vs predicted congestion are available in `notebooks/`.
- Heatmaps show **real traffic patterns vs current toll schedules**.

---

## Recommended Pricing Plan
Based on CART analysis, I propose a **simplified but optimized toll schedule**:
- **Mon-Fri:** 6am – 8pm (vs current 5am – 9pm)
- **Saturday:** 10am – 9pm (same as current)
- **Sunday:** 10am – 5pm (vs current 9am – 9pm)

This plan aligns tolling with true congestion peaks while avoiding unnecessary charges during off-peak hours.

---

## Project Structure
```
nyc-congestion-ml/
│
├── data/
│   └── nyc_congestion.csv
│
├── notebooks/
│   ├── Naive_Bayes.ipynb
│   ├── ANN.ipynb
│   ├── CART.ipynb
│   ├── Cluster.ipynb
│   ├── Isolation_Forest.ipynb
│   ├── KNN_final.ipynb
│   ├── Random_Forest.ipynb
│   └── SVM.ipynb
│
├── README.md
└── requirements.txt
```

---

## Installation & Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/nyc-congestion-ml.git
   cd nyc-congestion-ml
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Explore the notebooks:
   ```bash
   jupyter notebook notebooks/
   ```

---

## Future Work
- **Hyperparameter tuning** for all models.
- Integration with **real-time NYC traffic APIs**.
- **Flask/FastAPI deployment** of the best model for live congestion predictions.
- **Interactive dashboards** (Plotly/Streamlit).
