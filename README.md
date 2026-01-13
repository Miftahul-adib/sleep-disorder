# A Dual Pipeline Machine Learning Framework for Automated Multi Class Sleep Disorder Screening Using Hybrid Resampling and Ensemble Learning


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.67%25-brightgreen.svg)](https://github.com/yourusername/sleep-disorder-screening)

> **State-of-the-Art Solution for Multi-Class Sleep Disorder Classification**

This repository contains the official implementation of **"A Dual Pipeline Machine Learning Framework for Automated Multi-Class Sleep Disorder Screening Using Hybrid Resampling and Ensemble Learning"** (2026).

Our framework achieves **98.67% accuracy** on the Sleep Health and Lifestyle Dataset, surpassing current benchmarks through a novel dual-pipeline architecture that captures both linear and non-linear feature dependencies.

---

## Main paper:

- [Paper (arXiv)](https://arxiv.org/abs/xxxx.xxxxx)

---

## 👥 Authors

- **Md Sultanul Islam Ovi** - George Mason University
- **Muhsina Tarannum Munfa** - Metropolitan University, Sylhet
- **Miftahul Alam Adib** - Shahjalal University of Science and Technology
- **Syed Sabbir Hasan** - Shahjalal University of Science and Technology



---

## 🔍 Overview

Sleep disorders affect millions globally, yet accurate multi-class screening remains challenging. This framework addresses the problem of classifying patients into three categories:

- **Insomnia** (Class 0)
- **Healthy/None** (Class 1)
- **Sleep Apnea** (Class 2)



### Architecture Philosophy

We employ a **bifurcated processing approach**:
```
Input Data
    ├── Pipeline 1 (Statistical Path)
    │   ├── RobustScaler
    │   ├── Mutual Information Selection
    │   └── Linear Discriminant Analysis (LDA)
    │
    └── Pipeline 2 (Wrapper-Based Path)
        ├── MinMaxScaler
        ├── Boruta Feature Selection
        └── Autoencoder Compression
```

This dual approach ensures both linear separability and non-linear interaction capture.

---

## ✨ Key Features

### 1. **Dual-Pipeline Architecture**
- **Pipeline 1**: Optimizes for linear dependencies using statistical methods
- **Pipeline 2**: Captures non-linear interactions through wrapper-based selection

### 2. **Hybrid Resampling Strategy**
Addresses severe class imbalance (Healthy: 219, Apnea: 78, Insomnia: 77):
- **SMOTE**: Synthetic minority oversampling
- **Tomek Links**: Decision boundary cleaning

### 3. **Comprehensive Feature Engineering**
- 8 custom interaction features capturing physiological dependencies
- Tree-based importance ranking
- Cross-validated feature stability analysis

### 4. **Rigorous Validation**
- Stratified 8-fold cross-validation
- Wilcoxon Signed-Rank Test for statistical significance
- Multiple performance metrics (Accuracy, F1, Recall, Precision)

---

## 📊 Dataset

**Sleep Health and Lifestyle Dataset** (Sourced from Kaggle)
* **Samples**: 374 clinical entries.
* **Target Classes**: `None` (219), `Sleep Apnea` (78), `Insomnia` (77).
* **Features**: 13 attributes including Sleep Duration, BMI, Blood Pressure, and physical activity levels.



---
## 🛠 Methodology

### 1. Data Preprocessing

- Created 8 interaction features
- Tree-based importance analysis
- Information gain ranking

<img src="Images/Feature importance.png" alt="Feature importance" width="200">
### 2. Train-Test Split
- **80/20 stratified split** preserving class distribution

### 3. Class Balancing
We applied **SMOTETomek** to handle class imbalance, resampling the minority classes (Insomnia, Sleep Apnea) to align with the majority class (None).

| Classes | Original Count | Resampled Count |
| :--- | :---: | :---: |
| Insomnia (0) | 62 | 175 |
| None (1) | 175 | 173 |
| Sleep Apnea (2) | 62 | 171 |

### 4. Dual Pipeline Processing

#### Pipeline 1: Statistical
```python
# Scaling: RobustScaler (median/IQR normalization)
# Selection: Mutual Information (relevance filtering)
# Reduction: LDA (linear projection)
```

#### Pipeline 2: Wrapper-Based
```python
# Scaling: MinMaxScaler ([0,1] normalization)
# Selection: Boruta (shadow feature comparison)
# Reduction: Autoencoder (non-linear latent space)
```

### 5. Model Training & Validation
- **9 Classifiers**: Logistic Regression, K-Nearest Neighbors, Random Forest, XGBoost, Gradient Boosting, Extra Trees, AdaBoost, MLP Classifier, LightGBM
- **Hyperparameter Tuning**: RandomizedSearchCV
- **Cross-Validation**: Stratified 8-fold
- **Statistical Testing**: Wilcoxon Signed-Rank (p < 0.05)

---






## 📈 Results

### Top Performing Models

### Optimal performance metrics for Pipeline 1 utilizing statistical feature engineering and hybrid resampling strategies

| ML Model | Configuration | Accuracy | F1 Score | Recall | Precision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | RobustScaler + SMOTETomek | 96.000% | 94.379% | 94.798% | 94.737% |
| K-Nearest Neighbors | MI + SMOTETomek | 98.667% | 97.850% | 97.917% | 97.917% |
| Random Forest | LDA | 96.000% | 93.521% | 93.472% | 93.698% |
| XGBoost Model | MI + SMOTETomek | 98.667% | 97.850% | 97.917% | 97.917% |
| Gradient Boosting | RobustScaler | 96.000% | 94.177% | 93.611% | 94.815% |
| Extra Trees | RobustScaler | 97.333% | 95.658% | 95.556% | 96.296% |
| AdaBoost | MI | 97.333% | 95.694% | 95.694% | 95.694% |
| MLP Classifier | MI | 96.000% | 93.548% | 93.611% | 93.611% |
| LightGBM | MI + SMOTETomek | 98.667% | 97.850% | 97.917% | 97.917% |

<br>

### Optimal performance metrics for Pipeline 2 utilizing wrapper-based feature selection and non-linear dimensionality reduction

| ML Model | Configuration | Accuracy | F1 Score | Recall | Precision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | MinMaxScaler | 97.333% | 95.658% | 95.556% | 96.296% |
| K-Nearest Neighbors | Autoencoder | 93.333% | 92.324% | 93.422% | 91.799% |
| Random Forest | MinMaxScaler + SMOTETomek | 97.333% | 95.658% | 95.556% | 96.296% |
| XGBoost Model | MinMaxScaler + SMOTETomek | 97.333% | 95.694% | 95.694% | 95.694% |
| Gradient Boosting | Boruta | 96.000% | 93.521% | 93.472% | 93.698% |
| Extra Trees | Boruta + SMOTETomek | 98.667% | 97.840% | 97.778% | 98.039% |
| AdaBoost | Autoencoder | 94.667% | 93.780% | 94.179% | 93.669% |
| MLP Classifier | Autoencoder + SMOTETomek | 93.333% | 91.676% | 93.422% | 90.278% |
| LightGBM | MinMaxScaler | 97.333% | 95.694% | 95.694% | 95.694% |
### Performance Visualization

<p align="center">
  <img src="assets/confusion_matrix.png" width="45%" />
  <img src="assets/roc_curves.png" width="45%" />
</p>

### Statistical Validation

- **Wilcoxon Signed-Rank Test**: p = 0.00391 (W = 36.0)
- **Significance Level**: p < 0.05 ✓
- **Conclusion**: Performance gains are statistically significant

### Computational Efficiency

- **Inference Latency**: < 400ms per prediction
- **Training Time (Best Model)**: 1.04 seconds (KNN, Pipeline 1)
- **Suitable for**: Real-time clinical monitoring

---



## Sleep Disorder Classification - Comparative Analysis

#### Comparative analysis of the proposed framework against recent state of the art studies in sleep disorder classification

| Study | Dataset | Model | Accuracy |
|-------|---------|-------|----------|
| This study | Sleep Health & Lifestyle Dataset | Extra Trees Classifier | 98.667% |
| Ahadian et al.(2024)[1] | Multilevel Monitoring of Activity and Sleep in Healthy People | Long Short-Term Memory(LSTM) | 90% |
| Alshammari et al.(2024)[2] | Sleep Health & Lifestyle Dataset | Artificial Neural Networks(ANN) | 92.92% |
| Rahman et al.(2025)[6] | Sleep Health & Lifestyle Dataset | Gradient Boosting | 97.33% |
| Monowar et al.(2025)[7] | Sleep Health & Lifestyle Dataset | Ensemble Model | 96.88% |
| Hidayat et al.(2023)[17] | Sleep Health & Lifestyle Dataset | Random Forest | 88% |
| Dritsas et al.(2024)[21] | NHANES Dataset | SVM Polynomial | 91.44% |
| Han et al.(2023)[54] | Sleep Clinic of Samsung Medical Center | K-means Clustering | 91% |
| Satyavathi et al.(2025)[55] | Sleep Health & Lifestyle Dataset | Decision Tree | 96% |
| Alom et al.(2024)[56] | Sleep Health & Lifestyle Dataset | ANN Bagging ANN Boosting | 94.7% |
| Panda et al.(2025)[57] | Sleep Health & Lifestyle Dataset | Random Forest | 96% |
| Taher et al.(2024)[58] | Sleep Health & Lifestyle Dataset | Gradient Boosting | 93.8% |
| Zhu et al.(2023)[59] | Montreal Archive of Sleep Studies (MASS) | SwSleepNet | 86.7% |

**Improvement**: +1.34 percentage points over previous best

---

## 📄 Citation

If you use this code or methodology in your research, please cite:
```bibtex
@article{ovi2026dual,
  title={A Dual Pipeline Machine Learning Framework for Automated Multi Class Sleep Disorder Screening Using Hybrid Resampling and Ensemble Learning},
  author={Ovi, Md Sultanul Islam and Munfa, Muhsina Tarannum and Adib, Miftahul Alam and Hasan, Syed Sabbir},
  year={2026}
}
```



## 📝 License

This project is licensed under the MIT License 




---





---

<p align="center">
  <strong>⭐ Star this repo if you find it useful! ⭐</strong>
</p>
