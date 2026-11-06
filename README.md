# 🚢 Titanic – Machine Learning from Disaster

## 📌 Project Overview

This project focuses on developing a Machine Learning classification model to predict passenger survival on the Titanic using the [Kaggle Titanic – Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) dataset.

The objective is to predict whether a passenger survived the Titanic disaster (`Survived`: 0 = No, 1 = Yes) based on passenger information, including gender, ticket class, age, family relationships, and other attributes.

This project covers the end-to-end Machine Learning workflow, including Exploratory Data Analysis (EDA), feature selection, data preprocessing, model training, hyperparameter tuning, cross-validation, and prediction generation.

**Kaggle Notebook:** [Titanic – Machine Learning from Disaster](https://www.kaggle.com/code/hngvvan/titanic-machine-learning-from-disaster)

---

## 📊 Results & Evaluation

| Evaluation Method | Accuracy |
| :--- | :---: |
| 6-Fold Cross-Validation | 0.8174 |
| Hold-out Validation | 0.8000 |
| Kaggle Public Leaderboard | **0.78708** |

**Evaluation Metric: Accuracy**

Accuracy measures the proportion of correctly classified passenger survival predictions.

- **Cross-validation Accuracy:** 0.8174
- **Hold-out Accuracy:** 0.8000
- **Public Leaderboard Score:** 0.78708
- **Selected Model:** ExtraTreesClassifier

The cross-validation score represents the average performance across six validation folds, while the Kaggle Public Leaderboard score evaluates predictions on the competition's test dataset.

---

## 🛠️ Methodology & Implementation

### 1. Exploratory Data Analysis (EDA)

Conducted exploratory data analysis to understand passenger characteristics, feature distributions, and relationships with survival outcomes.

Key analysis areas:

- **Target Distribution:** Examined the distribution of survived and non-survived passengers.
- **Correlation Analysis:** Investigated relationships between numerical features using a correlation matrix.
- **Feature Analysis:** Explored passenger attributes such as gender, ticket class, age, and family size.
- **Missing Value Analysis:** Identified incomplete information, particularly in `Age`, `Cabin`, and `Embarked`.

### 2. Feature Selection

Selected relevant passenger attributes for model development while excluding `Name` and `Ticket`.

**Numerical Features:**
- `Pclass`
- `Age`
- `SibSp`
- `Parch`
- `Fare`

**Categorical Features:**
- `Sex`
- `Cabin`
- `Embarked`

This feature selection approach focuses on passenger characteristics and ticket-related information represented by the selected variables.

### 3. Data Preprocessing

Implemented a preprocessing architecture using Scikit-learn's `ColumnTransformer` within a `Pipeline` to handle numerical and categorical features separately.

**Numerical Features:**
- Applied `SimpleImputer(strategy='median')` to handle missing values.
- Used `StandardScaler` to standardize numerical feature scales.

**Categorical Features:**
- Applied `SimpleImputer(strategy='constant', fill_value='None')` to handle missing values.
- Used `OneHotEncoder(handle_unknown='ignore')` to transform categorical variables into numerical representations.
- Configured the encoder to handle previously unseen categories during prediction.

**Pipeline Architecture:**

The preprocessing transformations and classification model were integrated into a single Scikit-learn `Pipeline`.

This ensures that preprocessing steps are fitted independently within each training fold during cross-validation, helping prevent data leakage.

### 4. Model Development

**Selected Model: ExtraTreesClassifier**

Used `ExtraTreesClassifier` (Extremely Randomized Trees) as the core classification algorithm.

Extra Trees is an ensemble learning method that combines multiple randomized decision trees to improve classification performance.

Key characteristics:

- Builds an ensemble of randomized decision trees.
- Introduces additional randomness when selecting split thresholds.
- Captures non-linear relationships between passenger attributes and survival outcomes.
- Combines predictions from multiple trees to produce the final classification.

### 5. Hyperparameter Tuning

Applied Scikit-learn's `GridSearchCV` to search for an effective combination of model hyperparameters.

**Search Configuration:**
- Cross-validation: 6-fold
- Scoring metric: Accuracy

**Hyperparameter Search Space:**
- `n_estimators`
- `max_depth`
- `min_samples_leaf`
- `max_features`

**Best Parameters:**

| Parameter | Best Value |
| :--- | :---: |
| `n_estimators` | 350 |
| `max_depth` | 8 |
| `min_samples_leaf` | 3 |
| `max_features` | `sqrt` |

The best configuration was selected based on cross-validation accuracy.

### 6. Model Evaluation & Prediction

Evaluated the selected model using a hold-out validation set.

The evaluation process included:

- Measuring classification accuracy.
- Generating a `classification_report` to examine precision, recall, and F1-score.
- Comparing hold-out performance with cross-validation results.

After model evaluation, the trained pipeline was used to predict passenger survival on the competition's `test.csv` dataset.

The final predictions were exported to `submission.csv` following Kaggle's required submission format.

---

## 💻 Tech Stack

| Category | Technologies |
| :--- | :--- |
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Development Environment | Jupyter Notebook |
| Platform | Kaggle |

---

## 📁 Project Structure

```text
Titanic-Machine-Learning-from-Disaster/
│
├── Titanic - Machine Learning from Disaster.ipynb
│                                      # Main notebook: EDA, preprocessing,
│                                      # model training, tuning and evaluation
│
├── submission.csv                     # Generated Kaggle predictions
│
└── README.md                          # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <https://github.com/vuvanhung2005hd-lang/House-Prices-Advanced-Regression-Techniques.git>
cd Titanic-Machine-Learning-from-Disaster
```

### 2. Install Dependencies

Ensure that Python 3.8 or later is installed.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Download the Dataset

Download the competition dataset from Kaggle:

[Titanic – Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

Place the following files in the project directory:

- `train.csv`
- `test.csv`

### 4. Run the Notebook

```bash
jupyter notebook "Titanic - Machine Learning from Disaster.ipynb"
```

Run the notebook cells sequentially to reproduce the data analysis, preprocessing, hyperparameter tuning, model evaluation, and prediction workflow.

---

## 🎯 Key Learnings

Through this project, I gained practical experience in:

- Performing Exploratory Data Analysis on structured classification datasets.
- Analyzing missing values and relationships between passenger attributes.
- Selecting relevant numerical and categorical features.
- Building preprocessing pipelines using `ColumnTransformer` and `Pipeline`.
- Applying numerical imputation, feature scaling, and categorical encoding.
- Training ensemble classification models using Extra Trees.
- Optimizing model hyperparameters with `GridSearchCV`.
- Applying K-Fold Cross-Validation for model evaluation.
- Evaluating classification performance using Accuracy, Precision, Recall, and F1-score.
- Generating predictions and preparing submissions for Kaggle competitions.

---

