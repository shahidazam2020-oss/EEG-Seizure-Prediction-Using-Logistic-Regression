# 🧠 EEG Signal Classification Using Machine Learning

## 📌 Project Overview

This project investigates the use of **Machine Learning for EEG signal classification**. The main objective is to study how different preprocessing strategies, feature reduction techniques, regularization methods, and class imbalance techniques affect the performance of a Logistic Regression classifier.

The project uses EEG datasets containing multi channel brain signal measurements and evaluates different approaches for binary classification.

The experiments focus particularly on the **EEG Eye State Detection dataset**, where the target variable `eyeDetection` represents two classes:

`0` = Eyes closed
`1` = Eyes open

The study also loads and examines two additional EEG datasets for exploratory analysis and comparison.

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze EEG datasets using Python.
2. Perform basic data preprocessing and cleaning.
3. Standardize EEG features.
4. Apply denoising using rolling window smoothing.
5. Compare two feature processing pipelines.
6. Train Logistic Regression as the baseline classifier.
7. Investigate underfitting and overfitting.
8. Compare L1, L2, and Elastic Net regularization.
9. Evaluate class imbalance handling techniques.
10. Compare model performance using Accuracy, F1 Score, and PR AUC.
11. Identify the preprocessing and regularization strategy that provides the best performance.

---

## 📊 Datasets

Three EEG datasets are loaded in the notebook.

### 1. EEG Dataset

This dataset contains raw EEG features from multiple electrode channels.

After removing an unnecessary column, the dataset contains:

• **8,064 samples**
• **32 EEG features**

The features represent EEG electrode channels such as:

`Fp1`, `AF3`, `F3`, `F7`, `FC5`, `C3`, `T7`, `CP5`, `Cz`, `C4`, `T8`, `P4`, `P8`, `O2`, and others.

### 2. Epileptic Seizure Recognition Dataset

This dataset contains EEG signal measurements with **178 signal features** and a target variable `y`.

The original target contains multiple classes. For this experiment, the target is converted into a binary classification problem:

`1` = seizure class
`0` = non seizure classes

Dataset size:

• **11,500 samples**
• **178 signal features**
• **1 binary target**

### 3. EEG Eye State Dataset

The main dataset used for the classification experiments contains EEG measurements from 14 channels.

Dataset size:

• **14,980 samples**
• **14 EEG features**
• **1 target variable**

The target variable is:

`eyeDetection`

Class distribution:

| Class | Samples |
| ----- | ------: |
| 0     |   8,257 |
| 1     |   6,723 |

The dataset therefore contains a moderate class imbalance.

---

## 🛠️ Technologies and Libraries

The project was developed using Python and the following libraries:

| Technology       | Purpose                          |
| ---------------- | -------------------------------- |
| Python           | Programming language             |
| NumPy            | Numerical computation            |
| Pandas           | Data loading and manipulation    |
| Matplotlib       | Visualization                    |
| Scikit Learn     | Machine Learning                 |
| imbalanced learn | SMOTE and undersampling          |
| Jupyter Notebook | Development environment          |
| Kaggle           | Dataset and notebook environment |

---

## 🔬 Methodology

The project follows the workflow below:

```text
EEG Datasets
     │
     ▼
Data Loading
     │
     ▼
Data Cleaning
     │
     ▼
Missing Value Handling
     │
     ▼
Feature Standardization
     │
     ├─────────────────────┐
     │                     │
     ▼                     ▼
Pipeline A             Pipeline B
Denoising              Statistical Features
     │                     │
     ▼                     ▼
Variance Threshold       PCA
     │                     │
     └──────────┬──────────┘
                ▼
       Logistic Regression
                │
                ▼
          Model Evaluation
                │
       ┌────────┼─────────┐
       ▼        ▼         ▼
   Accuracy    F1      PR AUC
```

---

# 🔧 Data Preprocessing

## Missing Value Handling

Missing values are handled using mean imputation.

For example:

```python
df1 = df1.fillna(df1.mean())
df2 = df2.fillna(df2.mean())
X = X.fillna(X.mean())
```

Unnecessary columns are also removed where required.

---

## Standardization

The EEG features are standardized using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

This transforms the features to a common scale before applying further processing.

---

# 🧪 Experimental Pipelines

## Pipeline A

Pipeline A focuses on preserving the original EEG feature structure.

### Processing steps

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

Rolling window smoothing is applied using a window size of 3.

Variance Threshold is then used to remove features with very low variance.

Pipeline A retains:

**14 features**

---

## Pipeline B

Pipeline B performs statistical feature extraction followed by PCA.

### Processing steps

```text
Original EEG Features
        │
        ▼
Statistical Feature Extraction
        │
        ▼
Mean
Standard Deviation
Maximum
Minimum
Range
        │
        ▼
Standardization
        │
        ▼
PCA
        │
        ▼
Logistic Regression
```

The original 14 EEG measurements are reduced to five statistical features:

• Mean
• Standard deviation
• Maximum
• Minimum
• Range

PCA is then applied while retaining 95% of the variance.

The resulting representation contains:

**2 principal components**

---

# 🤖 Machine Learning Model

## Logistic Regression

Logistic Regression is used as the baseline classification model.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

The dataset is divided into training and testing sets using an 80:20 split with stratification.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

# 📈 Evaluation Metrics

The models are evaluated using:

### Accuracy

Measures the overall proportion of correctly classified samples.

### F1 Score

Provides a balance between precision and recall and is particularly useful when class distributions are not perfectly balanced.

### PR AUC

Measures the quality of the model's precision recall performance across classification thresholds.

---

# 📊 Experimental Results

## Pipeline Comparison

| Pipeline   |   Accuracy |  F1 Score |    PR AUC |
| ---------- | ---------: | --------: | --------: |
| Pipeline A | **58.01%** | **0.435** | **0.572** |
| Pipeline B |     55.31% |     0.210 |     0.489 |

### Observation

Pipeline A clearly performs better than Pipeline B.

Pipeline B compresses the original EEG representation into a small number of statistical features and then applies PCA. This causes information loss from the original multi channel EEG measurements.

The experiment therefore indicates that preserving the original EEG feature structure is more effective than aggressive feature compression for this dataset.

---

# ⚖️ Underfitting and Overfitting

The project also examines the effect of Logistic Regression regularization strength.

### Underfitting Model

A very small value of `C` is used:

```python
LogisticRegression(
    C=0.001,
    max_iter=1000
)
```

Result:

| Metric   |  Score |
| -------- | -----: |
| Accuracy | 56.74% |
| F1 Score |  0.238 |

The strong regularization restricts the model too much, resulting in underfitting.

### Overfitting Model

A very large value of `C` is used:

```python
LogisticRegression(
    C=1000,
    max_iter=1000
)
```

Result:

| Metric   |     Score |
| -------- | --------: |
| Accuracy |    64.25% |
| F1 Score | **0.554** |

The weaker regularization allows the model to fit the training data more closely.

---

# 🧩 Regularization Comparison

Three regularization strategies are evaluated:

• L1
• L2
• Elastic Net

## Results

| Regularization |   Accuracy |  F1 Score |
| -------------- | ---------: | --------: |
| **L1**         | **62.45%** | **0.518** |
| L2             |     58.01% |     0.435 |
| Elastic Net    |     57.34% |     0.426 |

### Best Result

**L1 regularization achieved the best performance among the three regularization approaches.**

L1 regularization can force some model coefficients toward zero, effectively performing feature selection. This can be useful when EEG features are correlated or contain redundant information.

---

# ⚖️ Class Imbalance Experiments

Three approaches are tested for handling class imbalance.

### 1. Class Weighting

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
```

### 2. SMOTE

Synthetic Minority Oversampling Technique is applied to the training data.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
```

### 3. Random Undersampling

Random undersampling reduces the number of samples from the majority class.

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
```

## Results

| Method        |  F1 Score |    PR AUC |
| ------------- | --------: | --------: |
| Class Weight  |     0.533 |     0.570 |
| SMOTE         |     0.532 | **0.571** |
| Undersampling | **0.536** |     0.568 |

### Observation

The three imbalance handling methods produce relatively similar results.

Undersampling achieves the highest F1 score, while SMOTE achieves the highest PR AUC.

Overall, the improvements are relatively small, suggesting that class imbalance has less influence on performance than the choice of feature representation and preprocessing strategy.

---

# 🏆 Key Findings

The experiments lead to several important observations:

### 1. Preserving EEG structure matters

Pipeline A performs considerably better than Pipeline B.

### 2. Aggressive feature compression can reduce performance

Reducing the EEG data to only a few statistical features results in information loss.

### 3. L1 regularization performs well

Among L1, L2, and Elastic Net, L1 produces the highest F1 score.

### 4. Class imbalance has a smaller effect

Class weighting, SMOTE, and undersampling produce only small differences.

### 5. Preprocessing has a major effect

The experimental results indicate that preprocessing and feature representation have a stronger influence on classification performance than the class imbalance strategies tested.

---

# 📁 Repository Structure

A recommended GitHub structure for this project is:

```text
EEG Signal Classification/
│
├── 📓 shahid-assignment(1).ipynb
│
├── 📂 datasets/
│   ├── features_raw.csv
│   ├── Epileptic Seizure Recognition.csv
│   └── EEG-Eye-State.csv
│
├── 📂 results/
│   ├── pipeline_comparison.png
│   ├── regularization_comparison.png
│   ├── class_imbalance_comparison.png
│   └── learning_curve.png
│
├── 📂 outputs/
│   └── model_results.csv
│
└── 📄 README.md
```

If the original datasets are too large for GitHub, keep them outside the repository and provide their Kaggle dataset references instead.

---

# 🚀 How to Run

## 1. Clone the repository

```bash
git clone https://github.com/yourusername/your-repository-name.git
```

## 2. Open the notebook

Open:

```text
shahid-assignment(1).ipynb
```

using Jupyter Notebook, JupyterLab, VS Code, or Kaggle.

## 3. Install the required libraries

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn
```

## 4. Update dataset paths

The original notebook uses Kaggle input paths.

For example:

```python
/kaggle/input/datasets/...
```

If running locally, update these paths to match your local `datasets` folder.

## 5. Run the notebook

Run the notebook cells from top to bottom to reproduce the preprocessing, experiments, evaluation metrics, and analysis.

---

# 📌 Main Notebook

The complete implementation, experiments, outputs, and analysis are available in:

**`shahid-assignment(1).ipynb`**

The notebook contains:

• Dataset loading
• Exploratory analysis
• Data preprocessing
• EEG signal denoising
• Feature selection
• Statistical feature extraction
• PCA
• Logistic Regression
• Train test evaluation
• Learning curve analysis
• Underfitting experiment
• Overfitting experiment
• L1 regularization
• L2 regularization
• Elastic Net
• Class weighting
• SMOTE
• Random undersampling
• Comparative analysis

---

# 📚 Academic Context

This project was developed as part of academic work in **Machine Learning and EEG signal analysis**.

It demonstrates how preprocessing decisions and model configuration can affect the performance of machine learning systems working with high dimensional biomedical signal data.

---

# 👨‍💻 Author

## Shahid Azam

**MS Computer Science Student**
Institute of Management Sciences, Peshawar, Pakistan

### Areas of Interest

• Artificial Intelligence
• Machine Learning
• Deep Learning
• Natural Language Processing
• Data Analysis
• Data Visualization
• EEG Signal Analysis

---

## ⭐ Project Highlights

```text
EEG Data Analysis
        +
Data Preprocessing
        +
Feature Engineering
        +
Machine Learning
        +
Regularization
        +
Class Imbalance Analysis
        =
EEG Classification Study
```

If you find this project useful, consider giving the repository a ⭐ on GitHub.
