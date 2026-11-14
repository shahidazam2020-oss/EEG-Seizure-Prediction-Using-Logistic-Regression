<div align="center">

# 🧠 EEG Signal Classification Using Machine Learning

### A Comparative Study of Logistic Regression, Feature Engineering, Regularization & Class Imbalance Handling

<p>
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Machine%20Learning-Classification-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Domain-EEG%20Signal%20Analysis-purple?style=for-the-badge">
  <img src="https://img.shields.io/badge/Research-Experimental-lightgrey?style=for-the-badge">
</p>

<p>
  <a href="#overview">Overview</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#methodology">Methodology</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#results">Results</a> •
  <a href="#installation">Installation</a>
</p>

**An experimental machine learning study exploring how preprocessing, feature representation, regularization, and class imbalance strategies influence EEG classification performance.**

[![GitHub Stars](https://img.shields.io/github/stars/shahidazam2020-oss/Data-Visualization?style=social)](https://github.com/shahidazam2020-oss/Data-Visualization/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/shahidazam2020-oss/Data-Visualization?style=social)](https://github.com/shahidazam2020-oss/Data-Visualization/forks)

</div>

---

## 📌 Project Overview

This project investigates the application of Machine Learning techniques to EEG (Electroencephalography) signal classification using **Logistic Regression**.

The primary objective is to evaluate how different data preprocessing strategies, feature engineering techniques, dimensionality reduction methods, regularization approaches, and class imbalance techniques affect classification performance.

The experiments examine EEG datasets and compare different feature-processing pipelines to understand their influence on predictive performance.

> **Research Focus:** Understanding the impact of feature representation and preprocessing decisions on machine learning classification of EEG signals.

---

## 🎯 Research Objectives

- Analyze EEG datasets using Python.
- Perform data cleaning and preprocessing.
- Handle missing values and standardize EEG features.
- Apply rolling-window smoothing for denoising.
- Compare alternative feature-processing pipelines.
- Implement Logistic Regression as the baseline classifier.
- Investigate underfitting and overfitting.
- Compare L1, L2, and Elastic Net regularization.
- Evaluate class imbalance handling strategies.
- Compare models using Accuracy, F1 Score, and PR AUC.
- Identify the most effective experimental configuration.

---

## 🗂️ Dataset Overview

**The project examines three EEG datasets.**

| Dataset | Samples | Features | Purpose |
|---|---:|---:|---|
| EEG Dataset | 8,064 | 32 | EEG signal exploration |
| Epileptic Seizure Recognition | 11,500 | 178 | Seizure vs. non-seizure classification |
| EEG Eye State | 14,980 | 14 | Eyes-open vs. eyes-closed classification |

### Dataset Details

**1. EEG Dataset**

**Contains measurements from multiple EEG electrode channels, including Fp1, AF3, F3, F7, FC5, C3, T7, CP5, Cz, C4, T8, P4, P8, and O2.**

**2. Epileptic Seizure Recognition Dataset**

**Contains 178 signal features and a target variable `y`.**

**For the binary classification experiment:**

- `1` – Seizure class
- `0` – Non-seizure classes

**3. EEG Eye State Dataset**

**The main classification dataset contains measurements from 14 EEG channels.**

Target variable: `eyeDetection`

- `0` – Eyes closed
- `1` – Eyes open

**Class distribution:**

| Class | Samples |
|---|---:|
| Eyes Closed (0) | 8,257 |
| Eyes Open (1) | 6,723 |

---

## 🛠️ Technology Stack

| Technology | Application |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computation |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning and preprocessing |
| Imbalanced-learn | SMOTE and undersampling |
| Jupyter Notebook | Experimental development |
| Kaggle | Dataset and notebook environment |

---

## 🔬 Research Methodology

**The experimental workflow consists of data preparation, feature processing, model training, and performance evaluation.**

```text
              EEG DATASETS
                   │
                   ▼
             DATA LOADING
                   │
                   ▼
             DATA CLEANING
                   │
                   ▼
          MISSING VALUE HANDLING
                   │
                   ▼
            STANDARDIZATION
                   │
          ┌────────┴────────┐
          ▼                 ▼
      PIPELINE A         PIPELINE B
          │                 │
    Rolling Smoothing  Statistical Features
          │                 │
    Variance Threshold      PCA
          │                 │
          └────────┬────────┘
                   ▼
          LOGISTIC REGRESSION
                   │
                   ▼
             MODEL EVALUATION
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
       Accuracy    F1     PR AUC
```

---

## ⚙️ Data Preprocessing

### Missing Value Handling

**Mean imputation is used to handle missing numerical values.**

```python
df = df.fillna(df.mean())
```

### Feature Standardization

**EEG features are standardized using `StandardScaler`.**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Signal Denoising

**Rolling-window smoothing is applied to reduce short-term fluctuations in the EEG measurements.**

---

## 🧪 Experimental Pipelines

### Pipeline A – Original EEG Feature Processing

**This pipeline focuses on preserving the original EEG feature representation.**

```text
Original EEG Features
        │
        ▼
Standardization
        │
        ▼
Rolling Window Smoothing
        │
        ▼
Variance Threshold
        │
        ▼
Logistic Regression
```

- Rolling window size: 3
- Feature selection: Variance Threshold
- Retained features: 14

### Pipeline B – Statistical Feature Extraction and PCA

**This pipeline extracts statistical information and applies dimensionality reduction.**

```text
Original EEG Features
        │
        ▼
Statistical Feature Extraction
        │
        ▼
Mean | Standard Deviation
Maximum | Minimum | Range
        │
        ▼
Standardization
        │
        ▼
PCA (95% Variance)
        │
        ▼
Logistic Regression
```

**The original 14 EEG measurements are transformed into five statistical features, followed by PCA.**

Final representation: **2 principal components**.

---

## 🤖 Machine Learning Model

### Logistic Regression

**Logistic Regression is used as the baseline classification algorithm.**

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
```

### Training and Testing

**The dataset is divided using an 80:20 stratified train-test split.**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## 📊 Model Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| F1 Score | Balance between precision and recall |
| PR AUC | Precision-recall performance across thresholds |

The evaluation emphasizes F1 Score and PR AUC alongside accuracy to provide a broader view of classification performance.

---

## 🧬 Experimental Investigations

### 1. Pipeline Comparison

| Pipeline | Accuracy | F1 Score | PR AUC |
|---|---:|---:|---:|
| Pipeline A | 58.01% | 0.435 | 0.572 |
| Pipeline B | 55.31% | 0.210 | 0.489 |

**Finding:** Pipeline A performs better than Pipeline B in all three reported metrics.

**This suggests that aggressive feature compression may discard useful information from the original EEG measurements.**

### 2. Underfitting and Overfitting

The regularization parameter `C` is investigated to understand its effect on model performance.

| Experiment | C | Accuracy | F1 Score |
|---|---:|---:|---:|
| Underfitting | 0.001 | 56.74% | 0.238 |
| Overfitting experiment | 1000 | 64.25% | 0.554 |

### 3. Regularization Comparison

Three regularization approaches are evaluated.

| Regularization | Accuracy | F1 Score |
|---|---:|---:|
| L1 | 62.45% | 0.518 |
| L2 | 58.01% | 0.435 |
| Elastic Net | 57.34% | 0.426 |

**Best reported approach: L1 Regularization**

L1 regularization can reduce less useful model coefficients toward zero, providing a form of feature selection.

### 4. Class Imbalance Handling

The following approaches are investigated:

- Class Weighting
- SMOTE (Synthetic Minority Oversampling Technique)
- Random Undersampling

| Method | F1 Score | PR AUC |
|---|---:|---:|
| Class Weighting | 0.533 | 0.570 |
| SMOTE | 0.532 | 0.571 |
| Undersampling | 0.536 | 0.568 |

**Observations:**

- Random undersampling achieved the highest F1 Score.
- SMOTE achieved the highest PR AUC.
- All three approaches produced relatively similar results.

---

## 🏆 Key Research Findings

| Research Area | Main Observation |
|---|---|
| Feature Representation | Pipeline A outperformed Pipeline B |
| Feature Compression | Aggressive compression reduced performance |
| Regularization | L1 achieved the best reported regularization results |
| Class Imbalance | The three tested strategies produced similar results |
| Preprocessing | Feature-processing decisions had a substantial influence on performance |

### Overall Conclusion

The experiments demonstrate that preprocessing and feature representation are important factors in EEG classification.

Preserving the original EEG feature structure produced better results than the statistical feature extraction and PCA pipeline tested in this study.

**Important:** This is an experimental machine learning project, not a clinically validated seizure-detection or diagnostic system.

---

## 📁 Repository Structure

The currently visible GitHub repository contains:

```text
EEG-Seizure-Prediction-Using-Logistic-Regression/
│
├── Major Assignment.rar
│
└── README.md
```

The compressed archive contains the project materials. The README describes the main implementation notebook as:

`shahid-assignment(1).ipynb`

---

## 🚀 Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/shahidazam2020-oss/EEG-Seizure-Prediction-Using-Logistic-Regression.git
```

### 2. Navigate to the Project

```bash
cd EEG-Seizure-Prediction-Using-Logistic-Regression
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn jupyter
```

### 4. Extract the Project Archive

Extract `Major Assignment.rar` to access the project materials.

### 5. Open the Notebook

Open the main notebook using Jupyter Notebook, JupyterLab, VS Code, or Kaggle.

### 6. Configure Dataset Paths

If running locally, update the original Kaggle dataset paths to match your local dataset locations.

### 7. Execute the Notebook

Run the notebook cells sequentially to reproduce the preprocessing, experiments, evaluation, and analysis.

---

## 📚 Skills Demonstrated

- EEG Signal Analysis
- Machine Learning
- Binary Classification
- Data Cleaning and Preprocessing
- Feature Engineering
- Feature Selection
- Principal Component Analysis (PCA)
- Logistic Regression
- Model Regularization
- Class Imbalance Handling
- Model Evaluation
- Experimental Comparison
- Python Programming

---

## 🔭 Future Improvements

Potential directions for extending this project include:

- Compare Logistic Regression with Random Forest, SVM, and Gradient Boosting.
- Explore advanced EEG signal-processing techniques.
- Investigate frequency-domain feature extraction.
- Evaluate additional feature-selection methods.
- Apply cross-validation for more robust performance estimation.
- Explore deep learning architectures for EEG classification.
- Include confusion matrices, ROC curves, and precision-recall curves.
- Evaluate performance on independent datasets.

---

## 👨‍💻 About the Developer

### Shahid Azam

**MS Computer Science | Artificial Intelligence**

Institute of Management Sciences, Peshawar, Pakistan

Interested in Artificial Intelligence, Machine Learning, Deep Learning, Natural Language Processing, and research-oriented software development.

### Connect With Me

- 💻 GitHub: [shahidazam2020-oss](https://github.com/shahidazam2020-oss)
- 🔗 LinkedIn: [Shahid Azam](https://www.linkedin.com/in/shahid-azam-mughal-787b58235/)
- 📚 ResearchGate: [Shahid Azam](https://www.researchgate.net/profile/Shahid-Azam-5)
- 🆔 ORCID: [0009-0001-6825-8863](https://orcid.org/0009-0001-6825-8863)
- 📧 Email: shahidazam2020@gmail.com

---

## 🤝 Contributions

Suggestions, improvements, and research discussions are welcome.

If you would like to contribute:

1. Fork this repository.
2. Create a new branch.
3. Implement your improvements.
4. Submit a pull request.

---

## 📜 License and Academic Use

This repository is intended for educational, academic, and experimental research purposes.

Please provide appropriate attribution when reusing the project or its materials. Check the repository for any applicable dataset licensing requirements.

---

<div align="center">

### ⭐ Support This Project

If you find this project useful, consider giving the repository a star.

**Explore • Learn • Experiment • Contribute**

🧠 *Advancing Machine Learning Research Through EEG Signal Analysis*

</div>
