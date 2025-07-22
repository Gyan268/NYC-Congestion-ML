# NYC Congestion Machine Learning Project

## Overview
This project analyzes NYC traffic congestion data and builds **binary classification models** to predict congestion levels (High vs Low) based on various features.  
The dataset was processed, cleaned, and used to train multiple machine learning models to evaluate performance.

---

## Models Implemented
The project includes eight different machine learning models, each in its own Jupyter Notebook:

- **Naive Bayes** (`notebooks/Naive_Bayes.ipynb`)
- **Artificial Neural Network (ANN)** (`notebooks/Artificial_Neural_Network_(ANN).ipynb`)
- **CART (Classification and Regression Trees)** (`notebooks/CART_(Classification_and_Regression_Trees).ipynb`)
- **Clustering** (`notebooks/Clustering.ipynb`)
- **Isolation Forest** (`notebooks/Isolation_Forest.ipynb`)
- **K-Nearest Neighbors (KNN)** (`notebooks/K-Nearest_Neighbors_(KNN).ipynb`)
- **Random Forest** (`notebooks/Random_Forest.ipynb`)
- **Support Vector Machine (SVM)** (`notebooks/Support_Vector_Machine_(SVM).ipynb`)

---

## Dataset
- The dataset is located in the `data/` folder: **`nyc_congestion.csv`**
- It includes features such as time, traffic volume, vehicle counts, and congestion levels.
- The target variable is binary (1 = High Congestion, 0 = Low Congestion).

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

3. Open the notebooks:
   ```bash
   jupyter notebook notebooks/
   ```

---

## Results
- Each model notebook contains **accuracy, confusion matrix, and performance metrics**.
- Comparison of models can help identify the best-performing approach for predicting congestion.

---

## Future Work
- Add hyperparameter tuning for each model.
- Integrate with real-time NYC traffic data APIs.
- Deploy the best model using Flask or FastAPI.

---

## Author
Developed by **[Your Name]**.

