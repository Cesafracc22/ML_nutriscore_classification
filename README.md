# Nutri-Score ML Prediction - OpenFoodFacts API

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

## Overview

This project implements a multi-class classification system for predicting Nutri-Score grades of food products based on their nutritional composistion and score. The Nutri-Score is a nutrition label (goes from a through e). Grade 'a' represents the best nutritional profile, while grade 'e' indicates products with worst characteristics.

The classification task uses nutritional data from the Open Food Facts database to train and evaluate five different machine learning algorithms. The project includes a complete data pipeline from raw data acquisition through preprocessing, model training, and evaluation. All models are trained with class balancing techniques and evaluation is performed on a stratified validation split to ensure reliable performance estimates across all Nutri-Score categories.

For all models, hyperparameter tuning is performed using GridSearchCV to optimize model performance. The best result for each algorithm is reported after a grid search over the relevant hyperparameter ranges.

## Quick Start

The project provides bash scripts for each step of the pipeline. (For windows, need to convert to .bat files)

### 1. Data Download and Preprocessing

```bash
./run-preprocessing.sh
```

This script performs the following operations:
- Downloads the Open Food Facts dataset (approximately 3GB)
- Filters products with complete nutritional information
- Samples 250,000 products for training
- Executes the 7-stage preprocessing pipeline
- Creates stratified train/validation/test splits (70%/15%/15%)
- Saves processed data to `data/splits/`

### 2. Model Training and Evaluation

```bash
./run-training-and-evaluation.sh
```

The script shows a menu to select from the five available models:
1. Logistic Regression
2. K-Nearest Neighbors
3. Support Vector Machine
4. Random Forest
5. XGBoost

After model selection, the script:
- Trains the selected model 
- Evaluates performance on the validation set
- Returns classification metrics and confusion matrix
- Saves the trained model to `models/trained/<model_name>/`

### 3. Manual Training (Alternative)

Can also run the training python file directly:

```bash
python scripts/train_model.py --model svm
```

Available model ids: `logistic_regression`, `knn`, `svm`, `random_forest`, `xgboost`

### 4. Model Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate_model.py \
    --model-path models/trained/svm/svm_v1.joblib \
    --model-type svm \
    --show-report \
    --show-confusion-matrix
```

## Installation

### Prerequisites
- Python 3.8+

### Setup

Clone the repository:
```bash
git clone https://github.com/djacoo/ml-nutriscore-prediction.git
cd ml-nutriscore-prediction
```

Create and activate a virtual environment:
```bash
python -m venv ml-predictor
source ml-predictor/bin/activate  # On Windows: ml-predictor\Scripts\activate
```

Install dependencies using one of the following methods:

**1: using pip with pyproject.toml**
```bash
pip install -e .
```

**2: using requirements.txt**
```bash
pip install -r requirements.txt
```

## Project Structure

```
ml-nutriscore-prediction/
├── data/
│   ├── raw/                        # Original downloaded dataset
│   ├── processed/                  # Subset with filtered products (250k)
│   └── splits/                     # Train/validation/test splits
│       ├── X_train.csv
│       ├── y_train.csv
│       ├── X_val.csv
│       ├── y_val.csv
│       ├── X_test.csv
│       └── y_test.csv
├── src/
│   ├── data/                       # Data loading utilities
│   ├── features/                   # Preprocessing pipeline components
│   └── models/                     # Model definitions & registry
│       ├── base_model.py           # Abstract base model class
│       ├── model_registry.py       # Model registry and lookup
│       ├── logistic_regression_model.py
│       ├── knn_model.py
│       ├── svm_model.py
│       ├── random_forest_model.py
│       └── xgboost_model.py
├── scripts/
│   ├── download_data.py            # Download Open Food Facts data
│   ├── train_model.py              # Model training script
│   ├── evaluate_model.py           # Evaluation/reporting script
│   └── tune_model.py               # Hyperparameter tuning script
├── models/
│   └── trained/                    # Saved trained models
│       ├── logistic_regression/
│       ├── knn/
│       ├── svm/
│       ├── random_forest/
│       └── xgboost/
├── docs/
│   ├── notebooks/                  # Jupyter notebooks (EDA, etc.)
│   │   ├── eda.ipynb
│   │   ├── model_evaluation.ipynb
│   │   └── preprocessing.ipynb
│   └── preprocessing_pipeline.md   # Detailed preprocessing pipeline docs
├── run-preprocessing.sh            # Bash script for preprocessing pipeline
└── run-training-and-evaluation.sh  # Bash script for model training & eval
```

## Dataset

The project uses data from Open Food Facts, an open database of food products with ingredients, nutritional information, and other metadata.

**Dataset Characteristics:**
- Source: Open Food Facts API
- Total products in database: >3 million
- Filtered dataset: 250,000 products with complete nutritional data
- Features: 15 nutritional attributes
  - Energy (kcal and kJ)
  - Macronutrients (fat, saturated fat, carbohydrates, sugars, proteins)
  - Micronutrients (salt, fiber, sodium)
  - Additional metrics (fruits/vegetables percentage, etc.)
- Target variable: Nutri-Score grade (a, b, c, d, e)
- Class distribution: Imbalanced (addressed via class weighting)

**Data Splitting:**
- Training set: 70% (175,000 products)
- Validation set: 15% (37,500 products)
- Test set: 15% (37,500 products)
- Stratification: Applied to maintain class proportions across splits

## Preprocessing Pipeline

The preprocessing pipeline goes with 7 sequential stages used to prepare raw nutritional data for the machine learning models. Each transformation is fitted on the training set only to prevent data leakage.

**Pipeline Stages:**
1. **Missing Value Imputation** - Median imputation for nutritional features
2. **Duplicate Removal** - Elimination of identical product entries
3. **Outlier Detection** - Domain-specific outlier handling for nutritional ranges
4. **Feature Engineering** - Creation of nutritional ratios and thresholds
5. **Adaptive Scaling** - StandardScaler
6. **Dimensionality Reduction** - PCA with 95% variance applied
7. **Validation** - Data integrity checks and range validation

The pipeline is implemented as a system where each stage can be configured independently. The documentation of each stage is to be found in [docs/preprocessing_pipeline.md](docs/preprocessing_pipeline.md).

## Models

All models inherit from a common `BaseModel` abstract class that defines a consistent interface for training, prediction, and evaluation. This system allows for easy addition of new models.

### 1. Logistic Regression

**Type:** Linear classifier with L2 regularization  
**Use case:** Baseline model

**Performance:**
- Validation accuracy: 70.8%
- F1-macro: 68.6%
- **Test accuracy: 70.1%** | **Test F1-macro: 68.0%**
- Training time: ~4s

### 2. K-Nearest Neighbors

**Type:** Instance-based learning with distance weighting  
**Use case:** Non-parametric baseline

**Performance:**
- Validation accuracy: 75.1%
- F1-macro: 72.8%
- **Test accuracy: 75.1%** | **Test F1-macro: 73.0%**
- Training time: ~42s

### 3. Support Vector Machine

**Type:** Kernel-based classifier with RBF kernel  
**Use case:** Best performing model

**Performance:**
- Validation accuracy: 84.2%
- F1-macro: 82.2%
- **Test accuracy: 84.0%** | **Test F1-macro: 82.1%**
- Training time: ~1700s

**Note:** SVM achieves the best performance but has the longest training time.

### 4. Random Forest

**Type:** Ensemble of decision trees  
**Use case:** Good balance of performance and speed

**Performance:**
- Validation accuracy: 79.2%
- F1-macro: 77.1%
- **Test accuracy: 79.3%** | **Test F1-macro: 77.2%**
- Training time: ~46s

**Characteristics:** Some overfitting observed (train accuracy: 89.1%)

### 5. XGBoost

**Type:** Gradient boosting with regularization  
**Use case:** Strong performance with efficient training

**Performance:**
- Validation accuracy: 80.0%
- F1-macro: 77.4%
- **Test accuracy: 80.4%** | **Test F1-macro: 78.0%**
- Training time: ~12s

## Model Comparison

| Model               | Val Accuracy | Val F1-Macro | Test Accuracy | Test F1-Macro | Training Time | Notes                                  |
|---------------------|-------------|--------------|---------------|---------------|---------------|----------------------------------------|
| SVM                 | 84.2%       | 82.2%        | **84.0%**     | **82.1%**     | long (~1700s) | Best performance                       |
| XGBoost             | 80.0%       | 77.4%        | 80.4%         | 78.0%         | short (~12s)  | Strong performance, fast training      |
| Random Forest       | 79.2%       | 77.1%        | 79.3%         | 77.2%         | medium (~46s) | Good balance of speed/accuracy         |
| KNN                 | 75.1%       | 72.8%        | 75.1%         | 73.0%         | short (~42s)  | Non-parametric baseline                |
| Logistic Regression | 70.8%       | 68.6%        | 70.1%         | 68.0%         | short (~4s)   | Linear baseline                        |

---

**Authors:** Jacopo Parretti VR536104, Cesare Fraccaroli VR533061
