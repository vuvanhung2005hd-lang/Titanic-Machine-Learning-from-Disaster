# 🚢 Titanic - Machine Learning from Disaster
Predicting passenger survival on the Titanic with a scikit-learn pipeline.

- 🏆 Kaggle competition: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- 📓 Kaggle notebook: [hngvvan/titanic-machine-learning-from-disaster](https://www.kaggle.com/code/hngvvan/titanic-machine-learning-from-disaster)

## 📈 Results

| Setting | Accuracy |
|---------|----------|
| Cross-validation (6-fold, best model) | 0.8174 |
| Hold-out validation | 0.80 |
| 🥇 Kaggle public leaderboard | **0.78708** |

The metric is **accuracy** on the 418 passengers in `test.csv`.

## 🎯 Overview

The goal is to predict whether a passenger survived the sinking of the Titanic
(`Survived`: 0 = No, 1 = Yes) from passenger data such as sex, ticket class, age and family size.

## 📂 Dataset

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 891 | Passenger data with the `Survived` label |
| `test.csv` | 418 | Passenger data without the label |
| `gender_submission.csv` | 418 | Example submission file |

| Column | Meaning |
|--------|---------|
| `Pclass` | Ticket class (1 = upper, 2 = middle, 3 = lower) |
| `Sex`, `Age` | Passenger sex and age |
| `SibSp` | Number of siblings / spouses aboard |
| `Parch` | Number of parents / children aboard |
| `Ticket`, `Fare` | Ticket number and fare paid |
| `Cabin` | Cabin number (mostly missing) |
| `Embarked` | Port of embarkation (C, Q, S) |

## 🛠️ Approach

1. 🔍 **EDA**: plotted the target distribution and the correlation matrix of numerical features.
2. 🧩 **Feature selection**: `Name` and `Ticket` are not used.
   Numerical features: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`.
   Categorical features: `Sex`, `Cabin`, `Embarked`.
3. ⚙️ **Preprocessing** (`ColumnTransformer` inside a `Pipeline`)
   - Numerical features: median imputation + `StandardScaler`
   - Categorical features: constant imputation (`"None"`) + `OneHotEncoder`
4. 🌲 **Model**: `ExtraTreesClassifier`.
5. 🎛️ **Hyperparameter tuning**: `GridSearchCV` (6-fold, scoring = accuracy) over
   `n_estimators`, `max_depth`, `min_samples_leaf` and `max_features`.

   Best parameters:

   | Parameter | Value |
   |-----------|-------|
   | `n_estimators` | 350 |
   | `max_depth` | 8 |
   | `min_samples_leaf` | 3 |
   | `max_features` | `sqrt` |

6. ✅ **Evaluation and prediction**: evaluated the best pipeline on the hold-out set
   (`classification_report`), then predicted `test.csv` and saved `submission.csv`.

## 🗂️ Project structure

```
.
├── Titanic - Machine Learning from Disaster.ipynb   # full notebook
├── train.csv
├── test.csv
├── submission.csv                                   # predictions for Kaggle
└── README.md
```

## 🚀 How to run

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook "Titanic - Machine Learning from Disaster.ipynb"
```

Download `train.csv` and `test.csv` from the
[Kaggle competition page](https://www.kaggle.com/competitions/titanic/data)
and place them in the same folder as the notebook.
Running all cells will create `submission.csv`.
(On Kaggle, the notebook reads the data from `/kaggle/input/competitions/titanic/`.)
